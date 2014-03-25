from setuptools import setup, find_packages

setup(
    name='Sound card power meter',
    version='0.1.0',
    packages = find_packages(),
    install_requires = ['pyaudio'],
    description='Record mains power demand',
    author='Jack Kelly',
    author_email='jack.kelly@imperial.ac.uk',
    url='https://github.com/JackKelly/snd_card_power_meter/',
    long_description=open('README.md').read(),
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    keywords='smartmeters power electricity energy analytics redd '
             'disaggregation nilm nialm'
)
