This fast paced hands on worskhop is designed to bootstrap your Deep Learning. It quickly on boards Machine Learning concepts like regression, classification,matrix factorization etc. It introduces algorithms like k Nearest Neighbors, k means, recommender systems etc. It brings in tools like python for quick coding,pandas and numpy for data munging, matplotlib for visualization, scikit-learn for ready made machine learning algorithms. It does so with real life use cases like predicting house sale prices, sentiment analysis using restaurant reviews; real life data like people wikipedia, adult income data etc. and lots of hands on coding. We look at deep learning, deep features and transfer learning in the context of image  recognition. We dive into intuition behind commonly popular algorithm of gradient descent, forward and backward propagation in neural networks. This approach helps imbibe the concepts effectively. We go onto implementing logistic regression as single layer neural network from scratch completely in python. Later on we implement generic multi layer neural network in tensorflow.

You can download the entire course content (follow along slides, data for hands on assignments, developed code for all hands on assignments) from github repository of https://github.com/sameermahajan/MLWorkshop During the course you will develop all the code outlined here from scratch under the guidance of the instructor.  Follow along the slides that has screenshots of hands on jupyter ipython notebooks and embedded URLs to their full listing on the gihub repository.

I hope that you continue referring to programs developed here for tools, technologies and techniques (3 Ts) even as you progress through your Deep Learning career! Good Luck!

You can develop and execute all the code with ready made tensorflow docker image from Google Container Registry. If you have windows machine, you can install virtualbox and deploy ubuntu VM on it. You can also install docker for Windows from: https://docs.docker.com/docker-for-windows/install/ You need docker installed on your ubuntu machine (refer: https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/). 

To run the container:

sudo docker run -it -p 8888:8888 gcr.io/tensorflow/tensorflow

It downloads docker image with pre installed  software for this course and runs jupyter server ready  to be connected. At the end it will  say something like:

	Copy/paste this URL into your browser when you connect for the first time,
		to login with a token:
			http://localhost:8888/?token=a61f16ecb110ea6c5dc73820184b0b53f30ff4184c9ab634

Type / copy-paste this URL in your browser to verify that you  can start a IPython notebook and happy coding!

(Note that above jupyter connectivity would also work across machines as long as there is network connectivity between them).

You can get the container id of the running docker container id by running:

sudo docker ps

You can enter the container (for say installing some custom package like scikit-surprise used by one of the programs on recommenders here) by running:

sudo docker exec -it (container id) bash
