# University of Madras Result Scraper

A Streamlit web application that automates the process of fetching student results from the University of Madras website. This tool accepts an Excel file with student register numbers and dates of birth, then scrapes the results and provides them in a downloadable Excel format. It also includes LangChain-powered Q&A functionality to analyze the result data.

## Features

- Upload Excel file with student details
- Automated scraping of student results from the University of Madras website
- Export all results to a downloadable Excel file
- Ask questions about the results data using natural language (powered by LangChain)
- Sample template provided for easy setup

## Installation

1. Clone this repository
2. Install the required dependencies:

```
pip install -r requirements.txt
```

3. Run the Streamlit app:

```
streamlit run app.py
```

## Usage

1. Prepare an Excel (.xlsx) file with columns:
   - `Register Number`: The student's register number
   - `Date of Birth`: The student's date of birth (in DD/MM/YYYY format)

2. Upload your Excel file through the web interface

3. Click "Start Scraping Results"

4. When the process completes, download the results Excel file

5. To use the Q&A functionality, provide your OpenAI API key when prompted

## Sample Questions

After scraping the results, you can ask questions like:

- How many students passed all subjects?
- What is the average grade for a specific subject?
- Which college had the highest pass rate?
- List all students who failed any subject

## Note

This tool is designed for educational purposes. Please ensure you have permission to access the student information you are processing.

## Requirements

- Python 3.8+
- Chrome browser (for Selenium WebDriver)
- OpenAI API key (for Q&A functionality)
