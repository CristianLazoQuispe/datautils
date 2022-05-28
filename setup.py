from distutils.core import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name='datautils',
    version='0.1demo',
    packages=['datautils',],
    author='Cristian Lazo Quispe',
    author_email='mecatronico.lazo@gmail.com',
    license='MIT',
    long_description=open('README.txt').read(),
    url='https://github.com/CristianLazoQuispe/datautils'
)