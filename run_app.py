
import uvicorn

from core import settings
from core.utils import setup_logging
from backend.app import create_app

main_app = create_app(create_custom_static_urls=True,)

if __name__ == "__main__":
    setup_logging()

    uvicorn.run(
        "run_app:main_app",
        host=settings.run.host,
        port=settings.run.port,
        workers=4,
        reload=True,
        # ssl_keyfile=settings.security.private_key_path,
        # ssl_certfile=settings.security.certificate_path
    )
