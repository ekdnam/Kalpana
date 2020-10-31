import setuptools

setuptools.setup(
	name = "flycatcher",
	version = "0.0.1",
	author = "Aditya Mandke",
	author_email = "ekdnam@gmail.com",
	description = "A research framework for state-of-the-art implementations of Generative Adversarial Networks for PyTorch",
	long_description = open("README.md", "r").read(),
	long_description_content_type  = "text/markdown",
	licences  = "MIT",
	url = "https://github.com/ekdnam/flycatcher",
	classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords = "GAN genrative adversarial network pytorch",
    install_requires = [
    	"numpy",
    	"torch>=1.6.0",
    	"torchvision>=0.7.0"
    ],
    python_requires=">=3.6",
    zip_safe = False
	)