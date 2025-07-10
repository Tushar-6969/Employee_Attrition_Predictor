import os
import google.generativeai as genai 

# set gemini api key (replace with urs)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# using free version model (this one works)
model = genai.GenerativeModel("gemini-1.5-flash")

def generate_recommendation(row):
    # build prompt with some details from user row
    prompt = f"""
    This employee is predicted to leave.

    Info:
    - Age: {row.get('Age', 'N/A')}
    - Job Role: {row.get('JobRole', 'N/A')}
    - Job Satisfaction: {row.get('JobSatisfaction', 'N/A')}
    - OverTime: {row.get('OverTime', 'N/A')}
    - Monthly Income: {row.get('MonthlyIncome', 'N/A')}
    - Years at Company: {row.get('YearsAtCompany', 'N/A')}

    Suggest 2 HR actions to retain this employee.
    """

    try:
        response = model.generate_content(prompt)  # gemini response
        return response.text.strip()  # final clean text
    except Exception as e:
        return f"Gemini Error: {str(e)}"  # show err if any
