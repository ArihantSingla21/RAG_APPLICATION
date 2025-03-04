import setuptools

def get_requirements(file_path:str)->list[str]:
    with open("requirements.txt", "r") as f:
        return f.read().splitlines()

setuptools.setup(
    name="RAG_Application",
    author="Arihant Singla",
    version="0.1.0",
    packages=setuptools.find_packages(),
    install_requires=get_requirements(file_path="requirements.txt"),
)