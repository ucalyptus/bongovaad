from setuptools import setup, find_packages

# Read the contents of requirements.txt file
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

# Read the README for the long description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='bongovaad',
    version='0.5.0',
    description='Bengali Speech Recognition Tool using Hugging Face Inference API',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='ucalyptus',
    author_email='',  # Add your email if you want
    url='https://github.com/ucalyptus/bongovaad',
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'bongovaad=bongovaad.transcriber:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Multimedia :: Sound/Audio :: Speech',
    ],
    keywords='bengali, speech recognition, whisper, youtube, transcription, subtitles, huggingface, api',
)
