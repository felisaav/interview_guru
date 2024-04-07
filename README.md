# AI Interview Guru
Welcome to AI Interview Guru, your ultimate tool for streamlining the job application process and enhancing interview preparation using cutting-edge artificial intelligence technologies.

## Main Models
1. ChatCompletion Model
   
The ChatCompletion model, powered by OpenAI's GPT-3.5, is used throughout AI Interview Guru to assist HR recruiters and job seekers in **Preprocessing job descriptions and resumes**, **parsing resumes to extract relevant insights**
  
2. Embedding Vector Model

This model, text-embedding-3-small, is tailored for generating embeddings, which are **numerical representations of textual data**. Embeddings capture the semantic meaning of words, phrases, or entire documents in a high-dimensional vector space.

## Formulas and Explanation
> Cosine Similarity Score

The cosine similarity score measures the cosine of the angle between two vectors in a multidimensional space. In the context of AI Interview Guru, it quantifies the similarity between the embeddings of a resume and a job description. The formula for cosine similarity is as follows:

\[
\frac{{|A| \cdot |B|}}{{A \cdot B}}
\]

Where:
- A and B are the embedding vectors of the resume and job description, respectively.
- |A| and |B| represent the magnitudes (or norms) of the vectors.

## Usage
JobMatchRecruiter: Upload resumes and job descriptions to match candidates with suitable positions. Receive detailed scoring and insights to make informed hiring decisions.

## Installation
To run AI Interview Guru locally, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies using pip install -r requirements.txt.
3. Set up your OpenAI API key as a secret variable in Streamlit.
4. Run the application using streamlit run AI_interview_checker.py.
5. Access the application through the provided URL.

## Contributing
Contributions to AI Interview Guru are welcome! If you have suggestions for new features, improvements, or bug fixes, please submit a pull request or open an issue on GitHub.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE.txt) file for details.
