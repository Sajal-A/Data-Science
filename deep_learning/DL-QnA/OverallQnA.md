# Fundamentals of Artificial Intelligence

**What is AI?**
- AI is a broad field that encompasses the development of intelligent systems capable of performing tasks that typically require human intelligence, such as perception, reasoning,
learning, problem-solving, and decision-making. AI serves as an umbrella term for various techniques and approaches, including machine learning, deep learning, and generative AI, among others.

**What is ML?**
- ML is a type of AI for understanding and building methods that make it possible for machines to learn.
These methods use data to improve computer performance on a set of tasks.

**What is DL?**
- Deep learning uses the concept of neurons and synapses similar to how our brain is wired. An example of a deep learning application is Amazon Rekognition,
which can analyze millions of images and streaming and stored videos within seconds.

**What is GenAI?**
- Generative AI is a subset of deep learning because it can adapt models built using deep learning, but without retraining or fine tuning.
- Generative AI systems are capable of generating new data based on the patterns and structures learned from training data.

## MACHINE LEARNING FUNDAMENTALS
Building a machine learning model involves data collection and preparation, selecting an appropriate algorithm, training the model on the prepared data, 
and evaluating its performance through testing and iteration.

**What are the different ways of representing training data?**
- The machine learning process starts with collecting and processing training data. Bad data is often called garbage in, garbage out, and
therefore an ML model is only as good as the data used to train it. Although data preparation and processing are sometimes a routine process,
it is arguably the most critical stage in making the whole model work as intended or ruining its performance.
- There are several different types of data used in training an ML model.
  - Let's understand the **Difference between labeled and unlabeled data.**
  - *Labeled Data* : Labeled data is a dataset where each instance or example is accompanied by a label or target variable that represents the desired output or classification. These labels are typically provided by human experts or obtained through a reliable process.
      - Example: In an image classification task, labeled data would consist of images along with their corresponding class labels (for example, cat, dog, car).
  - *Unlabeled Data* : Unlabeled data is a dataset where the instances or examples do not have any associated labels or target variables. The data consists only of input features, without any corresponding output or classification.
      - Example:  A collection of images without any labels or annotations
  - Based on the structural representation of data used for training are:
  - *Structured Data* : Structured data refers to data that is organized and formatted in a predefined manner, typically in the form of tables or databases with rows and columns. This type of data is suitable for traditional machine learning algorithms that require well-defined features and labels. The following are types of structured data.
      - `Tabular data`: This includes data stored in spreadsheets, databases, or CSV files, with rows representing instances and columns representing features or attributes.
      - `Time-series data`:  This type of data consists of sequences of values measured at successive points in time, such as stock prices, sensor readings, or weather data.
  - *Unstructured Data* : Unstructured data is data that lacks a predefined structure or format, such as text, images, audio, and video. This type of data requires more advanced machine learning techniques to extract meaningful patterns and insights.
     - `Text data`: This includes documents, articles, social media posts, and other textual data.
     - `Image data`: This includes digital images, photographs, and video frames.

**What are the different ML learning processes?**
- *Supervised Learning*: In supervised learning, the algorithms are trained on labeled data. The goal is to learn a mapping function that can predict the output for new, unseen input data.
- *Unsupervised Leanrning*: Unsupervised learning refers to algorithms that learn from unlabeled data. The goal is to discover inherent patterns, structures, or relationships within the input data.
- *Reinforcement Learning*: In reinforcement learning, the machine is given only a performance score as guidance and semi-supervised learning, where only a portion of training data is labeled. Feedback is provided in the form of rewards or penalties for its actions, and the machine learns from this feedback to improve its decision-making over time.

**What do you mean by inferencing? What are the main inferencing methodology used in ML?**
- After the model has been trained, it is time to begin the process of using the information that a model has learned to make predictions or decisions. This is called inferencing.
- *Batch inferencing* is when the computer takes a large amount of data, such as images or text, and analyzes it all at once to provide a set of results. This type of inferencing is often used for tasks like data analysis, where the speed of the decision-making process is not as crucial as the accuracy of the results.
- *Real-time inferencing* is when the computer has to make decisions quickly, in response to new information as it comes in. This is important for applications where immediate decision-making is critical, such as in chatbots or self-driving cars. The computer has to process the incoming data and make a decision almost instantaneously, without taking the time to analyze a large dataset.

## DEEP LEARNING
**Motivation from Human Brain**
At the core of deep learning are neural networks. Just like our brains have neurons that are connected to each other, neural networks have lots of tiny units called nodes that are connected together. These nodes are organized into layers. The layers include an input layer, one or more hidden layers, and an output layer.

**What is a Neural Network? Explain how intuitively it works.**
- At the core of deep learning are neural networks. Just like our brains have neurons that are connected to each other, neural networks have lots of tiny units called nodes that are connected together. These nodes are organized into layers. The layers include an input layer, one or more hidden layers, and an output layer.
- When we show a neural network many examples, like data about customers who bought certain products or used certain services, it figures out how to identify patterns by adjusting the connections between its nodes. It's like the nodes are talking to each other and slowly figuring out the patterns that separate different types of customers.
- When a neural network learns to recognize these patterns from the examples, it can then look at data for completely new customers that it has never seen before and still make predictions about what they might buy or how they might behave.

**In Which areas deep learning enhance the results?**
- *Computer Vision (CV)* : Computer vision is a field of artificial intelligence that makes it possible for computers to interpret and understand digital images and videos. Deep learning has revolutionized computer vision by providing powerful techniques for tasks such as image classification, object detection, and image segmentation.
  - Key points to understand about computer vision:
  - Computer vision is accomplished by using large numbers of images to train a model.
  - Image classification is a form of computer vision in which a model is trained with images that are labeled with the main subject of the image (in other words, what it's an image of) so that it can analyze unlabeled images and predict the most appropriate label - identifying the subject of the image.
  - Object detection is a form of computer vision in which the model is trained to identify the location of specific objects in an image.
  - There are more advanced forms of computer vision - for example, semantic segmentation is an advanced form of object detection where, rather than indicate an object's location by drawing a box around it, the model can identify the individual pixels in the image that belong to a particular object.
  - You can combine computer vision and language models to create a multi-modal model that combines computer vision and generative AI capabilities.
  - Common Applications of CV include:
      - Auto-captioning or tag-generation for photographs.
      - Visual search.
      - Monitoring stock levels or identifying items for checkout in retail scenarios.
      - Security video monitoring.
      - Authentication through facial recognition.
      - Robotics and self-driving vehicles.

- *Natural Language Processing (NLP)* : Natural language processing (NLP) is a branch of artificial intelligence that deals with the interaction between computers and human languages. Deep learning has made significant strides in NLP, making possible tasks such as text classification, sentiment analysis, machine translation, and language generation.
  - Key points to understand about NLP:
  - NLP capabilities are based on models that are trained to do particular types of text analysis.
  - While many natural language processing scenarios are handled by generative AI models today, there are many common text analytics use cases where simpler NLP language models can be more cost-effective.
  - Common NLP tasks include:
      - Entity extraction - identifying mentions of entities like people, places, organizations in a document
      - Text classification - assigning document to a specific category.
      - Sentiment analysis - determining whether a body of text is positive, negative, or neutral and inferring opinions.
      - Language detection - identifying the language in which text is written.
  - Common applications of NLP:
      - Analyzing document or transcripts of calls and meetings to determine key subjects and identify specific mentions of people, places, organizations, products, or other entities.
      - Analyzing social media posts, product reviews, or articles to evaluate sentiment and opinion.
      - Implementing chatbots that can answer frequently asked questions or orchestrate predictable conversational dialogs that don't require the complexity of generative AI.

## GENERATIVE AI FUNDAMENTALS
**What is Generative AI?**
- Generative AI is a branch of AI that enables software applications to generate new content; often natural language dialogs, but also images, video, code, and other formats.
- The ability to generate content is based on a language model, which has been trained with huge volumes of data - often documents from the Internet or other public sources of information.
- Generative AI models encapsulate semantic relationships between language elements (that's a fancy way of saying that the models "know" how words relate to one another), and that's what enables them to generate a meaningful sequence of text.
- There are large language models (LLMs) and small language models (SLMs) - the difference is based on the volume of data and the number of variables in the model. LLMs are very powerful and generalize well, but can be more costly to train and use. SLMs tend to work well in scenarios that are more focused on specific topic areas, and usually cost less.
  
- Common uses of generative AI include:
  - Implementing chatbots and AI agents that assist human users.
  - Creating new documents or other content (often as a starting point for further iterative development)
  - Automated translation of text between languages.
  - Summarizing or explaining complex documents.
  
**Foundational Models**
- `Amazon Bedrock provides access to a choice of high-performing FMs from leading AI companies like AI21 Labs, Anthropic, Cohere, Meta, Mistral AI, Stability AI, and Amazon.
With these FMs as a foundation, you can further optimize their outputs with prompt engineering, fine-tuning, or RAG.`
- *Let's understand the concept of FMs*
- Generative AI is powered by models that are pretrained on internet-scale data, and these models are called foundation models (FMs).  With FMs, instead of gathering labeled data for each model and training multiple models as in traditional ML, you can adapt a single FM to perform multiple tasks. These tasks include text generation, text summarization, information extraction, image generation, chatbot interactions, and question answering. FMs can also serve as the starting point for developing more specialized models.

**Explain the lifecycle of a Foundational Model**
The foundation model lifecycle is a comprehensive process that involves several stages, each playing a crucial role in developing and deploying effective and reliable foundation models.










  







