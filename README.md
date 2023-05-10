# Automatic Target Recognition


In this project, we developed a classification system using 3D CNN and SVM to recognize targets in 3D images obtained from ranged sensory systems. Our objective was to accurately classify images into four categories: saline, clay, rubber, and non-targets. This system falls under the category of Automatic Target Recognition (ATR) algorithms, which are highly attractive for security and surveillance applications due to their ability to interpret large amounts of data more efficiently than humans.

To begin the project, we conducted an exhaustive review of the literature to identify the most suitable algorithms for target recognition in 3D images. We selected 3D CNN as our primary algorithms due to their high accuracy and ability to handle complex data. We trained and tested these algorithms on a dataset of 3D images consisting of saline, clay, rubber, and non-targets.

We designed our 3D CNN architecture to capture the spatiotemporal features of the 3D images. Our CNN consisted of multiple 3D convolutional layers followed by max-pooling layers to reduce the spatial dimensions of the feature maps. We also added dropout layers to prevent overfitting. The output of the CNN was flattened,then which made the final prediction.

To evaluate the performance of our system, we used metrics such as testing accuracy accuracy, precision, recall. We also compared our results with other state-of-the-art algorithms for target recognition. Our experiments showed that the 3D CNN and SVM algorithms were indeed viable options for target recognition in 3D images. The highest testing accuracy being 92% for 3D CNN model and 85% for our SVM.

In conclusion, we developed an efficient target recognition system using 3D CNN and SVM algorithms. Our system demonstrated high accuracy and outperformed other state-of-the-art algorithms in target recognition. Our system has the potential to be used in various security and surveillance applications, including border control, airport security, and military surveillance.


Conference Paper: https://docs.google.com/document/d/1yiskczlku2ky9L5zSWceyGq46t8FFos3NFFBgXQHClk/edit?usp=sharing
