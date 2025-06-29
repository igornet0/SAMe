# import asyncio
# from sqlalchemy.future import select
# from sqlalchemy.orm import joinedload, selectinload
# from sqlalchemy import update
# import json

# from core import data_helper
# from backend.MMM import AgentManager
# from backend.Dataset import LoaderTimeLine, DatasetTimeseries
# from backend.train_models import Loader as TrainLoader

# from core.database import (db_helper, AgentTrain, Agent, AgentFeature,
#                            orm_get_feature_by_id,
#                            orm_get_agent_by_id,
#                            orm_get_timeseries_by_coin,
#                            orm_get_data_timeseries)

# async def start_process_train(train_data: dict):
#    async with db_helper.get_session() as session:

#         # Получаем процесс
#         query = select(AgentTrain).filter(AgentTrain.id == train_data["id"])
#         query = query.options(joinedload(AgentTrain.coins))

#         query = query.options(selectinload(AgentTrain.agent)
#                               .selectinload(Agent.features)
#                               .selectinload(AgentFeature.feature_value))

#         result = await session.execute(query)
#         process = result.scalars().first()
        
#         if process and process.status == "start":
#             print(f"Запуск процесса ID: {process.id}")
            
#             # Обновляем статус - начат
#             process.set_status("train")
#             agent = await orm_get_agent_by_id(session, process.agent_id)
#             agent.set_status("train")
#             agent.active = False

#             await session.commit()

#             # filename = data_helper["data"] / "t.json"
#             indecaters = {}

#             for feature in process.agent.features:
#                 parameters = {value.feature_name: value.value for value in feature.feature_value}
#                 feature_t = await orm_get_feature_by_id(session, feature.feature_id)
#                 indecaters[feature_t.name] = parameters

#             config = {"agents": [
#                 {
#                     "type": process.agent.type,
#                     "name": process.agent.name,
#                     "timetravel": process.agent.timeframe,
#                     "data_normalize": True,
#                     "mod": "train",
#                     "indecaters": indecaters
#                 }
#             ]}

#             async def load_loader(coin):
#                 tm = await orm_get_timeseries_by_coin(session, coin,
#                                                                 timeframe=process.agent.timeframe)
#                 data = await orm_get_data_timeseries(session, tm.id)
                
#                 dt = DatasetTimeseries(data)

#                 return dt
            
#             def filter_func(x):
#                 if x["open"] != "x" and isinstance(x["open"], str):
#                     print("ERORR !!!!!! - ", x)
#                     return True
#                 return x["open"] != "x"

#             loaders = [LoaderTimeLine(await load_loader(coin), 200,
#                                       filter_func, timetravel=process.agent.timeframe) for coin in process.coins]

#             agent_manager = AgentManager(config=config)
#             trin_loader = TrainLoader()

#             loader = trin_loader.load_agent_data(loaders, agent_manager.get_agents(), process.batch_size, False)

#             for data in loader:
#                 print(data)

#             # trin_loader._train_single_agent(agent_manager.get_agents(), loaders,
#             #                                 epochs=process.epochs,
#             #                                 batch_size=process.batch_size,
#             #                                 base_lr=process.learning_rate,
#             #                                 weight_decay=process.weight_decay,
#             #                                 patience=7,
#             #                                 mixed=True,
#             #                                 mixed_precision=True)

#             # print(agent_manager.get_agents())

#             # with open(filename, "w") as f:

#             #     json.dump({"coins": [coin.name for coin in process.coins],
#             #                 "agent": process.agent.id,
#             #                "features": features_data}, f)
            
#             # Здесь ваша логика запуска процесса (асинхронная)
#             # Например, вызов внешнего API или запуск асинхронного кода
#             # await asyncio.sleep(600)  # Имитация длительной задачи

#             print(f"Процесс ID: {process.id} завершен")
            
#             # Обновляем статус - завершен
#             process.set_status("stop")
#             agent.set_status("open")
#             agent.active = True
#             await session.commit()