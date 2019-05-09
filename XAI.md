# Weld & Bansal 2018 
## [The Challenge of Crafting Intelligible Intelligence](https://arxiv.org/pdf/1803.04263.pdf)

Most computer-based produced behavior is alien (can fail in unexpected ways). Complex systems that exceeds human abilities to verify. We can neither trust nor control system behavior that we do not understand. 

Seek AI systems that:
1. It is clear what factors caused the system's action, allowing users to predict how changes to the situation would have led to alternative behaviors
2. Permits effective control of the AI by enabling interaction

Their Approach to solving this problem:
1) Ensuring that the underlying reasoning or learned model is inherently interpretable (e.g. by learning a linear model over a small number of well-understoof features)
2) If it is necessary to use an inscrutable model, such as complex neural networks or deep-lookahead search, then mapping this complex system to a simplier, explanatory model for understanding and control

Provides transparency and veracity: a user can see what the model is doing

Problem: Interpretable methods may not performa as well as more complex ones

Mapping approach can apply to best performing technique but its explanation inherently differs from the way the AI system actually operates.

Answer: Make the explanation system interactive so users can drill down until they are satisfied with their understanding

**The key challenge for designing intelligle AI is communicating a complex computation process to a human.** This requires interdisciplinary skills, including HCI as well as AI and machine learning expertise. 

## Why Intelligibility Matters
### AI May Have The Wrong Objective
Simplified objective function such as accuracy combined with historically biased training data may cause uneven performance for different groups. Intelligibility empowers users ability to determine if an AI is right	for the right reasons.

### AI May Be Using Inadequate Features
Models extract any information they can from features but are susceptible to correlations. An intelligible model allows humans to spot these issues and correct them, e.g. by adding additional features.

### Distributional Drift
Deployed models may perform poorly in the wild, when a difference exists between the distribution which was used during training and that encountered during deployment
Deployment distribution may change over time, even from the act of deployment
-Adversarial domains: Spam detection, online ad pricing, and search engine optimization
Intelligibility helps users determine when models are failing to generalize

### Facilitating User Control
If users understand why the AI performed an undesired action, they can better issue instructions that will lead to improved future behavior

### User Acceptance
Users are happier with and more likely to accept algorithmic decisions if they are accomplanied by an explanation.

### Improving Human Insight
Intelligible models greatly facilitate the process of human understanding

### Legal Imperatives
Auditing situations (AI-specific error)to assess liability, requires understanding the model's decisions
  
# [Ribeiro, Singh, & Guestrin 2016](https://arxiv.org/pdf/1602.04938v1.pdf)
## “Why Should I Trust You?” Explaining the Predictions of Any Classifier
If the users do not trust a model or a prediction, they will not use it
(1) Trusting a prediction - user trusts an individual prediction sufficiently to take some action based on it
(2) Trusting a model - user trust a model to behave in reasonable ways if deployed

Solution: explanation for prediciton as a solution to 1st trust problem
			selecting multiple such predictions as a solution to the 2nd trust problem
Explanations: presenting textual or visual artifacts that provide qualitative 
understanding of the relationships between the instance's components and the model's 
prediction
Interpretable: provide qualitative understanding between joint values of input
				variables and the resulting predicted response values
Explanations should be easy to understand (not necessarily true for the models features)

# [Lundberg & Lee 2017](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf)
## A Unified Approach to Interpreting Model Predictions
Tension between accuracy and interpretability
Understanding how these methods relate and when one method is preferable to another
is still lacking

# [Ribeiro, Singh, & Guestrin 2018](https://homes.cs.washington.edu/~marcotcr/aaai18.pdf)
## Anchors: High-Precision Model-Agnostic Explanations

# [Springer & Whittaker 2018](https://arxiv.org/ftp/arxiv/papers/1812/1812.03220.pdf)
## What Are You Hiding? Algorithmic Transparency and User Perceptions.
Lack of transparency may lead users to accept output from algorithms that are simply random
Results around the effects of algorithmic transparency have been mixed
	Lim and Dey found that increased transparency can make users question the algorithm when its correct, impairing the user experience
	Bunt, Lount, and Lauzon found that users may feel that these explanations simply cause
		additional processing without offering real value

Transparency removed the correlation between expectation violation and accuracy

# [Adadi & Berrada 2018](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8466590)
## Peeking Inside the Black-Box: A Survey on Explainable Artificial Intelligence (XAI)

AI algorithms suffer from opacity, that it is difficult to get insight into their internal mechanism of work, especially Machine Learning algorithms. Which further compound the problem, because entrusting important decisions to a system that cannot explain itself presents obvious dangers.

To address this issue, Explainable Artificial Intelligence (XAI) proposes to make a shift towards more transparent AI. It aims to create a suite of techniques that produce more explainable models whilst maintaining high performance levels.

XAI is a research field that aims to make AI systems results more understandable to humans. According to DARPA [16], XAI aims
to ‘‘produce more explainable models, while maintaining a high level of learning performance (prediction accuracy); and enable human users to understand, appropriately, trust, and effectively manage the emerging generation of artificially intelligent partners’’.

The goal of enabling explainability in ML, as stated by FAT∗ [4], ‘‘is to ensure that algorithmic decisions as well as any data driving those decisions can be explained to end-users and other stakeholders in non-technical terms’’.

### Complexity Methods
The complexity of a machine-learning model is directly related to its interpretability. Generally, the more complex the model, the more difficult it is to interpret and explain. Thus, the most straightforward way to get to interpretable AI/ML would be to design an algorithm that is inherently and intrinsically interpretable. Many papers support this classic approach.

### Scoop Related Methods
Interpretability implies understanding an automated model, this supports two variations according to the scoop of interpretability: understanding the entire model behavior or understanding a single prediction. In the studied literature, contributions are made in both directions. Accordingly, we distinguish between two subclasses: (i) Global interpretability and (ii) Local interpretability

### Model Related Methods
Another important way to classify model interpretability techniques is whether they are model agnostic, meaning they can be applied to any types of ML algorithms, or model specific, meaning techniques that are applicable only for a single type or class of algorithm.
