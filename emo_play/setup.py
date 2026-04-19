from setuptools import setup, find_packages

setup(
    name="emo_play",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "python-dotenv>=0.19.0",
        "deepface>=0.0.75",
        "opencv-python>=4.5.3",
        "pydantic>=1.8.2",
        "numpy>=1.21.0",
        "hsemotion>=0.3",
        "facenet-pytorch>=2.5.2",
    ],
)