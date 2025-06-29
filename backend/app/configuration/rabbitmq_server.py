import aio_pika
import json
from core import settings
import logging

logger = logging.getLogger(__name__)

class AsyncRabbitMQ:
    _connection = None
    _channel = None

    # Унифицированные аргументы для ВСЕХ очередей
    queue_arguments: dict = {
        "x-message-ttl": 30000,            # TTL 30 секунд
        "x-dead-letter-exchange": "dlx",    # DLX имя должно быть одинаковым
        "x-max-priority": 10                # Добавляем приоритеты
    }

    async def get_connection(self) -> aio_pika.RobustConnection:
        """Создает или возвращает существующее соединение"""
        if self._connection is None or self._connection.is_closed:
            self._connection = await aio_pika.connect_robust(
                host=settings.rbmq.host,
                port=settings.rbmq.port,
                login=settings.rbmq.user,
                password=settings.rbmq.password,
            )
        return self._connection

    async def get_channel(self) -> aio_pika.Channel:
        """Создает или возвращает существующий канал"""
        if self._channel is None or self._channel.is_closed:
            connection = await self.get_connection()
            self._channel = await connection.channel()

        return self._channel

    async def send_message(self, queue: str, message: dict):
        """Асинхронная отправка сообщения в очередь"""
        channel = await self.get_channel()
        
        # Используем единый метод объявления с обработкой ошибок
        await self._declare_queue(
            channel,
            queue,
            durable=True,
            arguments=self.queue_arguments  # Унифицированные аргументы
        )
        
        await channel.default_exchange.publish(
            aio_pika.Message(
                body=json.dumps(message).encode(),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT
            ),
            routing_key=queue,
        )

    async def _declare_queue(self, channel: aio_pika.Channel, queue_name: str, **kwargs) -> aio_pika.Queue:
        """Безопасное объявление очереди с обработкой ошибок"""
        try:
            return await channel.declare_queue(queue_name, **kwargs)
        except aio_pika.exceptions.ChannelPreconditionFailed as e:
            logger.error(f"Queue precondition failed: {e}. Deleting and recreating queue {queue_name}")
            # Удаляем очередь и создаем заново с правильными параметрами
            await channel.queue_delete(queue_name)
            return await channel.declare_queue(queue_name, **kwargs)

    async def consume_messages(self, queue: str, callback: callable, prefetch_count: int = 1):
        channel = await self.get_channel()
        await channel.set_qos(prefetch_count=prefetch_count)
        
        # Используем те же аргументы, что и в send_message
        queue_obj = await self._declare_queue(
            channel,
            queue,
            durable=True,
            arguments=self.queue_arguments  # Унифицированные аргументы
        )
        
        async with queue_obj.iterator() as queue_iter:
            async for message in queue_iter:
                try:
                    async with message.process():
                        body = message.body.decode()
                        data = json.loads(body)
                        await callback(data)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    # При ошибке отправляем сообщение в DLX
                    await message.nack(requeue=False)

    async def setup_dlx(self):
        channel = await self.get_channel()
        await channel.declare_exchange("dlx", aio_pika.ExchangeType.FANOUT)

    async def close(self):
        """Закрывает соединение с RabbitMQ"""
        if self._channel and not self._channel.is_closed:
            await self._channel.close()
        if self._connection and not self._connection.is_closed:
            await self._connection.close()

# Создаем экземпляр для использования
rabbit = AsyncRabbitMQ()
