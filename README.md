# AI Interview Guru
Welcome to AI Interview Guru, your ultimate tool for streamlining the job application process and enhancing interview preparation using cutting-edge artificial intelligence technologies.

## Main Models
1. ChatCompletion Model
The ChatCompletion model, powered by OpenAI's GPT-3.5, is used extensively throughout AI Interview Guru to assist HR recruiters and job seekers in various tasks, including:

Preprocessing job descriptions and resumes.
Providing detailed analyses and feedback on resumes.
Matching candidates with suitable job positions based on skills, education, and experience.
Evaluating the quality of resumes and job descriptions.
Generating insights and suggestions for resume improvement.
2. Embedding Vector Model
The Embedding Vector model is utilized to generate embeddings for text data, enabling the comparison of resumes and job descriptions based on semantic similarity. This model calculates the cosine similarity score between two embeddings, allowing for accurate matching and scoring of resumes against job descriptions.

## Formulas and Explanation
Cosine Similarity Score
The cosine similarity score measures the cosine of the angle between two vectors in a multidimensional space. In the context of AI Interview Guru, it quantifies the similarity between the embeddings of a resume and a job description. The formula for cosine similarity is as follows:

Cosine Similarity
=
�
⋅
�
∥
�
∥
∥
�
∥
Cosine Similarity= 
∥A∥∥B∥
A⋅B
​
 

Where:

�
A and 
�
B are the embedding vectors of the resume and job description, respectively.
⋅
⋅ denotes the dot product of the two vectors.
∥
�
∥
∥A∥ and 
∥
�
∥
∥B∥ represent the magnitudes (or norms) of the vectors.
Usage
JobMatchRecruiter: Upload resumes and job descriptions to match candidates with suitable positions. Receive detailed scoring and insights to make informed hiring decisions.

Read Me: Stay updated on the latest features and developments in the application. Learn how to use each feature effectively and find answers to frequently asked questions.

## Installation
To run AI Interview Guru locally, follow these steps:

Clone the repository to your local machine.
Install the required dependencies using pip install -r requirements.txt.
Set up your OpenAI API key as a secret variable in Streamlit.
Run the application using streamlit run AI_interview_checker.py.
Access the application through the provided URL.
Contributing
Contributions to AI Interview Guru are welcome! If you have suggestions for new features, improvements, or bug fixes, please submit a pull request or open an issue on GitHub.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
