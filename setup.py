import setuptools

requirements = [
	"numpy",
	"torch>="
]
with open("README.md", "r") as fh:
	long_description = fh.read()

setuptools.setup(
	name = "flycatcher",
	version = "0.0.1a",
	author = "Aditya Mandke",
	author_email = "ekdnam@gmail.com",
	description = "A research framework for state-of-the-art implementations of Generative Adversarial Networks for PyTorch",
	licences  = "MIT",
	url = "https://github.com/ekdnam/flycatcher",

	)