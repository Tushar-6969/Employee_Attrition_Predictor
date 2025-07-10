import google.generativeai as genai

# Set your Gemini API key
genai.configure(api_key="AIzaSyCXToxk8Nfinvfi1rAX_rm8lVHwZ1fhmoo")

# Use a free plan-compatible model
model = genai.GenerativeModel("gemini-1.5-flash")

def generate_recommendation(row):
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
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Gemini Error: {str(e)}"
