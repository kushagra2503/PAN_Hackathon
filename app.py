import streamlit as st
import pandas as pd
import time
import re
import os
from datetime import datetime
import traceback
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.document_loaders import DataFrameLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
import io
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Check if GROQ_API_KEY is in environment variables
groq_api_key = os.getenv("GROQ_API_KEY")
print(f"GROQ_API_KEY: {groq_api_key}")

# Initialize session state to store dataframes between tabs
if 'df_successful' not in st.session_state:
    st.session_state.df_successful = pd.DataFrame()
    print("Initialized empty df_successful in session state")
if 'df_failed' not in st.session_state:
    st.session_state.df_failed = pd.DataFrame()
    print("Initialized empty df_failed in session state")

st.set_page_config(page_title="UoM Result Scraper", page_icon="📊", layout="wide")

st.title("University of Madras Result Scraper")
st.write("Upload an Excel file with student register numbers and dates of birth to fetch their results")

# Function to set up the WebDriver
def setup_driver(browser_type="chrome", headless=True):
    try:
        if browser_type == "chrome":
            chrome_options = Options()
            if headless:
                chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            return driver
        
        elif browser_type == "firefox":
            from selenium.webdriver.firefox.options import Options as FirefoxOptions
            firefox_options = FirefoxOptions()
            if headless:
                firefox_options.add_argument("--headless")
            firefox_options.add_argument("--window-size=1920,1080")
            
            service = Service(GeckoDriverManager().install())
            driver = webdriver.Firefox(service=service, options=firefox_options)
            return driver
        
        else:
            st.error(f"Unsupported browser type: {browser_type}")
            return None
            
    except Exception as e:
        st.error(f"Error setting up WebDriver: {str(e)}")
        return None

# Function to scrape results for a single student
def scrape_result(driver, reg_no, dob, max_retries=3):
    """
    Scrape the result for a single student from the University of Madras result website.
    Args:
        driver: Selenium WebDriver instance
        reg_no: Registration number of the student
        dob: Date of birth of the student
        max_retries: Maximum number of retries for scraping
    
    Returns:
        tuple: (result_dict, success_boolean)
            result_dict: Dictionary containing either result data or error information
            success_boolean: True if scraping was successful, False otherwise
    """
    retries = 0
    while retries < max_retries:
        try:
            # Navigate to the result page
            driver.get("https://egovernance.unom.ac.in/results/ugresult.asp")
            time.sleep(2)
            
            # Take screenshot of initial page for debugging
            if not os.path.exists("debug_info"):
                os.makedirs("debug_info")
            driver.save_screenshot(f"debug_info/initial_page_{reg_no}.png")
            
            # Fill the registration number - try multiple approaches
            try:
                reg_input = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.NAME, "regno"))
                )
            except TimeoutException:
                try:
                    # Try by ID
                    reg_input = driver.find_element(By.ID, "regno")
                except NoSuchElementException:
                    try:
                        # Try by CSS selector for input with name or id containing 'reg'
                        reg_input = driver.find_element(By.CSS_SELECTOR, "input[name*='reg'], input[id*='reg']")
                    except NoSuchElementException:
                        try:
                            # Try by finding inputs and looking for the first text input
                            inputs = driver.find_elements(By.XPATH, "//input")
                            reg_input = next((inp for inp in inputs if inp.get_attribute("type") == "text"), None)
                        except NoSuchElementException:
                            # Take screenshot for debugging
                            driver.save_screenshot(f"debug_info/form_not_found_{reg_no}.png")
                            with open(f"debug_info/page_source_{reg_no}.html", "w", encoding="utf-8") as f:
                                f.write(driver.page_source)
                            return {"Error": "Could not find registration number input field"}, False
            
            if reg_input is None:
                # Take screenshot for debugging
                driver.save_screenshot(f"debug_info/form_not_found_{reg_no}.png")
                with open(f"debug_info/page_source_{reg_no}.html", "w", encoding="utf-8") as f:
                    f.write(driver.page_source)
                return {"Error": "Could not find registration number input field"}, False
            
            # Clear and fill registration number input
            reg_input.clear()
            reg_input.send_keys(reg_no)
            
            # Fill Date of Birth - try multiple approaches
            dob_input = None
            
            # Method 1: Try by name
            try:
                dob_input = driver.find_element(By.NAME, "dob")
            except NoSuchElementException:
                pass
                
            # Method 2: Try by ID
            if dob_input is None:
                try:
                    dob_input = driver.find_element(By.ID, "dob")
                except NoSuchElementException:
                    pass
            
            # Method 3: Try by placeholder
            if dob_input is None:
                try:
                    dob_input = driver.find_element(By.CSS_SELECTOR, "input[placeholder*='date'], input[placeholder*='birth'], input[placeholder*='dob']")
                except NoSuchElementException:
                    pass
            
            # Method 4: Try by finding the label and then the adjacent input
            if dob_input is None:
                try:
                    dob_labels = driver.find_elements(By.XPATH, "//label[normalize-space()='Date of Birth']")
                    if dob_labels:
                        for label in dob_labels:
                            try:
                                # Try to find the input associated with this label
                                label_for = label.get_attribute("for")
                                if label_for:
                                    dob_input = driver.find_element(By.ID, label_for)
                                    break
                                else:
                                    # Try to find an input that's a sibling or child of the label's parent
                                    parent = label.find_element(By.XPATH, "./..")
                                    inputs = parent.find_elements(By.XPATH, "//input")
                                    if inputs:
                                        dob_input = inputs[0]
                                        break
                            except:
                                continue
                except:
                    pass
            
            # Method 5: Last resort - try to find the second input on the form
            if dob_input is None:
                try:
                    inputs = driver.find_elements(By.XPATH, "//input")
                    if len(inputs) >= 2:
                        dob_input = inputs[1]  # Assume the second input is DOB
                except:
                    pass
            
            if dob_input is None:
                driver.save_screenshot(f"debug_info/dob_not_found_{reg_no}.png")
                with open(f"debug_info/page_source_dob_{reg_no}.html", "w", encoding="utf-8") as f:
                    f.write(driver.page_source)
                return {"Error": "Could not find date of birth input field"}, False
            
            # Clear and fill date of birth input
            dob_input.clear()
            dob_input.send_keys(dob)
            
            # Find and click the submit button
            submit_button = None
            
            # Try different methods to find submit button
            button_locators = [
                (By.XPATH, "//input[@type='submit']"),
                (By.XPATH, "//button[@type='submit']"),
                (By.XPATH, "//input[@value='Submit']"),
                (By.XPATH, "//button[normalize-space()='Submit']"),
                (By.XPATH, "//input[@value='Get']"),
                (By.XPATH, "//button[normalize-space()='Get']"),
                (By.XPATH, "//input[@class='btn']"),
                (By.XPATH, "//button[@class='btn']"),
                # More generic approaches
                (By.XPATH, "//input[@type='button']"),
                (By.XPATH, "//button"),
                (By.XPATH, "//input[@type='image']"),
                (By.XPATH, "//input[contains(@value, 'ubmit')]"),
                (By.XPATH, "//input[contains(@value, 'earch')]"),
                (By.XPATH, "//input[contains(@value, 'ook')]"),
                (By.XPATH, "//input[contains(@value, 'et')]"),
                (By.XPATH, "//input[contains(@onclick, 'submit')]"),
                (By.CSS_SELECTOR, "form button"),
                (By.CSS_SELECTOR, "form input[type='submit']"),
                (By.CSS_SELECTOR, "form input[type='button']"),
                (By.CSS_SELECTOR, ".btn"),
                (By.CSS_SELECTOR, ".button")
            ]
            
            for locator in button_locators:
                try:
                    elements = driver.find_elements(*locator)
                    if elements:
                        submit_button = elements[0]
                        break
                except:
                    continue
            
            # If still no button found, try a fallback approach - look for any clickable element that might be a submit button
            if submit_button is None:
                try:
                    # Get all inputs and buttons
                    all_inputs = driver.find_elements(By.XPATH, "//input")
                    all_buttons = driver.find_elements(By.XPATH, "//button")
                    
                    # Check inputs first
                    for input_elem in all_inputs:
                        input_type = input_elem.get_attribute("type")
                        input_value = input_elem.get_attribute("value")
                        
                        # If it looks like a button or submit, use it
                        if input_type in ["button", "submit", "image"] or (input_value and len(input_value) < 15):
                            submit_button = input_elem
                            break
                    # If still not found, try any button
                    if submit_button is None and all_buttons:
                        submit_button = all_buttons[0]
                except:
                    pass
            
            if submit_button is None:
                # Create debug directory if it doesn't exist
                os.makedirs("debug_info", exist_ok=True)
                driver.save_screenshot(f"debug_info/submit_not_found_{reg_no}.png")
                # Save page HTML for debugging
                with open(f"debug_info/page_source_submit_{reg_no}.html", "w", encoding="utf-8") as f:
                    f.write(driver.page_source)
                return {"Error": "Could not find submit button"}, False
            
            # Click the submit button
            submit_button.click()
            time.sleep(3)  # Wait for results to load
            
            # Check for common error messages
            error_messages = [
                "Invalid Register Number",
                "Invalid Date of Birth",
                "No Results Found",
                "Record not found"
            ]
            
            for error_msg in error_messages:
                if error_msg.lower() in driver.page_source.lower():
                    return {
                        "Error": f"Website returned error: {error_msg}",
                        "Register Number": reg_no,
                        "Date of Birth": dob
                    }, False
            
            # Take screenshot of results
            driver.save_screenshot(f"debug_info/results_{reg_no}.png")
            
            # Extract result data
            results = {}
            results["Register Number"] = reg_no
            results["Date of Birth"] = dob
            
            # Try to find the student name
            try:
                # Try different methods to find student name
                name_element = None
                name_selectors = [
                    "//td[contains(text(), 'Name') or contains(text(), 'NAME') or contains(text(), 'name')]/following-sibling::td",
                    "//th[contains(text(), 'Name') or contains(text(), 'NAME') or contains(text(), 'name')]/following-sibling::td", 
                    "//tr[contains(.,'Name') or contains(.,'NAME') or contains(.,'name')]/td[2]",
                    "//label[contains(text(), 'Name') or contains(text(), 'NAME') or contains(text(), 'name')]/following-sibling::*",
                    "//div[contains(text(), 'Name') or contains(text(), 'NAME') or contains(text(), 'name')]/following-sibling::*",
                    "//td[contains(text(), 'Candidate') or contains(text(), 'CANDIDATE')]/following-sibling::td"
                ]
                
                # First attempt with exact selectors
                for selector in name_selectors:
                    try:
                        elements = driver.find_elements(By.XPATH, selector)
                        if elements:
                            for element in elements:
                                text = element.text.strip()
                                # Filter out common institution names or empty values
                                if (text and 
                                    len(text) > 3 and 
                                    "university" not in text.lower() and 
                                    "madras" not in text.lower() and
                                    "institution" not in text.lower() and
                                    "college" not in text.lower()):
                                    name_element = element
                                    break
                            if name_element:
                                break
                    except:
                        continue
                
                # If still not found, try looking for the typical positioning of names in result pages
                if not name_element:
                    try:
                        # Try to find tables with potential student info
                        tables = driver.find_elements(By.TAG_NAME, "table")
                        for table in tables:
                            rows = table.find_elements(By.TAG_NAME, "tr")
                            for row in rows:
                                cells = row.find_elements(By.TAG_NAME, "td")
                                if len(cells) >= 2:
                                    # Check if first cell contains something like "Name" or "Student"
                                    first_cell = cells[0].text.lower().strip()
                                    if "name" in first_cell or "student" in first_cell or "candidate" in first_cell:
                                        second_cell = cells[1].text.strip()
                                        if (second_cell and 
                                            "university" not in second_cell.lower() and 
                                            "madras" not in second_cell.lower() and
                                            len(second_cell) > 3):
                                            name_element = cells[1]
                                            break
                            if name_element:
                                break
                    except:
                        pass
                
                # If name element found, extract text
                if name_element:
                    student_name = name_element.text.strip()
                    # Final verification - if the name looks valid, use it
                    if (student_name and 
                        "university" not in student_name.lower() and 
                        "madras" not in student_name.lower() and
                        len(student_name) > 3):
                        results["Student Name"] = student_name
                    else:
                        results["Student Name"] = "Name extraction failed"
                else:
                    # Alternative method: look for elements with certain formatting
                    bold_elements = driver.find_elements(By.XPATH, "//b | //strong")
                    for element in bold_elements:
                        text = element.text.strip()
                        if (len(text) > 3 and 
                            ":" not in text and 
                            "university" not in text.lower() and 
                            "madras" not in text.lower() and
                            text.upper() != text):  # Likely a name
                            results["Student Name"] = text
                            break
                    else:
                        results["Student Name"] = "Name extraction failed"
            except Exception as e:
                results["Student Name"] = "Name extraction failed"
                print(f"Error extracting name: {str(e)}")
            
            # Find subject data (codes, names, marks)
            subject_results = {}
            
            # Look for tables
            tables = driver.find_elements(By.TAG_NAME, "table")
            for table in tables:
                try:
                    rows = table.find_elements(By.TAG_NAME, "tr")
                    for row in rows:
                        cells = row.find_elements(By.TAG_NAME, "td")
                        if len(cells) >= 3:  # Assuming at least: subject code, name, marks
                            subject_code = None
                            subject_name = None
                            marks = []
                            
                            # Check first cell for subject code
                            first_cell_text = cells[0].text.strip()
                            if first_cell_text and len(first_cell_text) <= 15:  # Subject codes are usually short
                                subject_code = first_cell_text
                                
                                # Check second cell for subject name if we found a code
                                if len(cells) > 1:
                                    subject_name = cells[1].text.strip()
                                
                                # Get all remaining cells as potential mark fields
                                for i in range(2, len(cells)):
                                    mark_text = cells[i].text.strip()
                                    marks.append(mark_text)
                                
                                # Only add if we have a subject code
                                if subject_code:
                                    # Store all marks as positional data
                                    for i, mark in enumerate(marks):
                                        subject_results[f"{subject_code}_{i}"] = mark
                except Exception as e:
                    continue
            
            # Add subject results to main results
            results.update(subject_results)
            
            # If we have subject data, consider it a success
            if subject_results:
                return results, True
            else:
                # Additional check: see if we're still on the input form (meaning submission failed)
                if "submit" in driver.page_source.lower() and ("regno" in driver.page_source.lower() or "dob" in driver.page_source.lower()):
                    # We're still on the form - retry
                    retries += 1
                    continue
                else:
                    # We're on a different page but couldn't extract data
                    return {
                        "Error": "Could not extract subject data from results page",
                        "Register Number": reg_no,
                        "Date of Birth": dob
                    }, False
        
        except Exception as e:
            # Handle any other exceptions
            retries += 1
            error_message = str(e)
            
            # If we've reached max retries, return the error
            if retries >= max_retries:
                return {
                    "Error": f"Exception: {error_message}",
                    "Register Number": reg_no,
                    "Date of Birth": dob
                }, False
            
            # Otherwise try again
            time.sleep(2)
    
    # If we've exhausted all retries
    return {
        "Error": "Maximum retries exceeded",
        "Register Number": reg_no,
        "Date of Birth": dob
    }, False

# Function to validate the input Excel file
def validate_excel_file(df):
    validation_errors = []
    
    # Check for required columns
    required_columns = ['Register Number', 'Date of Birth']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        validation_errors.append(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Check for empty values
    if 'Register Number' in df.columns and df['Register Number'].isnull().any():
        validation_errors.append("Some register numbers are missing")
    
    if 'Date of Birth' in df.columns and df['Date of Birth'].isnull().any():
        validation_errors.append("Some dates of birth are missing")
    
    # Check date format (if dates are stored as strings)
    if 'Date of Birth' in df.columns:
        date_col = df['Date of Birth']
        for i, dob in enumerate(date_col):
            if isinstance(dob, str):
                # Check if in DD/MM/YYYY format
                parts = dob.split('/')
                if len(parts) != 3 or not all(part.isdigit() for part in parts):
                    validation_errors.append(f"Row {i+1}: Date of birth '{dob}' is not in DD/MM/YYYY format")
    
    return validation_errors

# Function to process results for export
def process_results_for_export(all_results):
    """Process scraped results to generate a properly formatted Excel export like the user's example."""
    # Separate successful and failed results
    successful_results = []
    failed_results = []
    
    for result in all_results:
        if 'Error' in result:
            failed_results.append(result)
        else:
            successful_results.append(result)
    
    # Process successful results to create a dynamic DataFrame
    if successful_results:
        # First extract all unique subject codes from the results
        all_subject_codes = set()
        all_subject_positions = {}  # To track the order of subject codes/scores
        
        # First pass: collect all unique subject codes
        for result in successful_results:
            # Extract all keys that look like subject codes (format: CODE_N)
            subject_code_pattern = re.compile(r'^([A-Za-z0-9]+)_(\d+)$')
            
            for key in result.keys():
                match = subject_code_pattern.match(key)
                if match:
                    subject_code = match.group(1)
                    position = int(match.group(2))
                    
                    all_subject_codes.add(subject_code)
                    
                    # Track the positions seen for each subject code
                    if subject_code not in all_subject_positions:
                        all_subject_positions[subject_code] = set()
                    all_subject_positions[subject_code].add(position)
        
        # Convert to sorted list for consistent column ordering
        sorted_subject_codes = sorted(list(all_subject_codes))
        
        # Second pass: create rows with properly aligned subject data
        student_info = []
        for result in successful_results:
            # Basic student information as first columns
            student_row = {
                'NAME': result.get('Student Name', ''),
                'REG NO': result.get('Register Number', ''),
                'DOB': result.get('Date of Birth', '')
            }
            
            # For each subject code, create columns for each position observed
            for subject_code in sorted_subject_codes:
                # Get all positions for this subject
                positions = sorted(list(all_subject_positions.get(subject_code, [])))
                
                for position in positions:
                    key = f"{subject_code}_{position}"
                    column_name = subject_code
                    
                    # If multiple positions for this code, make each a separate column
                    if len(positions) > 1:
                        column_name = f"{subject_code}_{position}"
                    
                    # Set value if it exists in results
                    if key in result:
                        student_row[column_name] = result[key]
                    else:
                        student_row[column_name] = ""
            
            student_info.append(student_row)
        
        # Create DataFrame with all columns aligned
        df_successful = pd.DataFrame(student_info)
        
        # Create DataFrame for failed results
        if failed_results:
            df_failed = pd.DataFrame(failed_results)
        else:
            df_failed = pd.DataFrame()
        
        return df_successful, df_failed
    else:
        return pd.DataFrame(), pd.DataFrame(failed_results) if failed_results else pd.DataFrame()

# Function to get result for a single student
def get_result(regno, dob, browser_choice, headless, delay, debug_mode):
    """Get result for a single student."""
    # Initialize driver using the setup_driver function
    try:
        driver = setup_driver(browser_choice, headless)
        
        if driver is None:
            return {"Error": "Failed to initialize browser driver"}, False
        
        # Call the scrape_result function with the driver, registration number, and dob
        result, success = scrape_result(driver, regno, dob)
        
        # Add a delay to avoid overloading the site if needed
        if delay:
            time.sleep(delay)
        
        # Check if the result was successful
        if success:
            st.success(f"Successfully retrieved result for {regno}")
        else:
            st.error(f"Failed to retrieve result for {regno}: {result.get('Error', 'Unknown error')}")
            # If in debug mode, show any saved debugging information
            if debug_mode:
                debug_folder = "debug_info"
                if os.path.exists(debug_folder):
                    # Find all debug files for this registration number
                    debug_files = [f for f in os.listdir(debug_folder) if regno in f]
                    if debug_files:
                        st.write("Debug information available:")
                        for file in debug_files:
                            file_path = os.path.join(debug_folder, file)
                            if file.endswith(".png"):
                                st.image(file_path, caption=file)
                            elif file.endswith(".html"):
                                with open(file_path, "r", encoding="utf-8") as f:
                                    st.download_button(f"Download {file}", f.read(), file)
        
        # Close the browser
        driver.quit()
        
        return result, success
    
    except Exception as e:
        st.error(f"Error processing {regno}: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return {"Error": str(e), "Register Number": regno, "Date of Birth": dob}, False

# Create tabs for different functionality
tab1, tab2 = st.tabs(["Web Scraping", "Q&A"])

# Web Scraping tab
with tab1:
    st.header("Student Results Scraper")
    st.write("Upload a CSV or Excel file with Register Numbers and Dates of Birth, or enter them manually.")
    
    # Input method options
    input_method = st.radio("Choose Input Method", ["Upload File", "Manual Entry"])
    
    # Initialize lists for registration numbers and dates of birth
    reg_numbers = []
    dobs = []
    
    if input_method == "Upload File":
        uploaded_file = st.file_uploader("Choose a file", type=["xlsx", "csv"])
        
        if uploaded_file is not None:
            # Process the uploaded file
            if uploaded_file.name.endswith(".csv"):
                # Read CSV file
                df = pd.read_csv(uploaded_file)
            else:
                # Read Excel file
                df = pd.read_excel(uploaded_file)
            
            # Validate the Excel file
            validation_errors = validate_excel_file(df)
            if validation_errors:
                st.error("The uploaded file has the following errors:")
                for error in validation_errors:
                    st.error(error)
                reg_numbers = []
                dobs = []
            else:
                # Extract registration numbers and dates of birth
                reg_numbers = [str(x) for x in df["Register Number"].tolist()]
                dobs = [str(x) for x in df["Date of Birth"].tolist()]
                
                # Display preview of the data
                st.subheader("Data Preview")
                st.dataframe(df[["Register Number", "Date of Birth"]].head())
                st.write(f"Total entries: {len(reg_numbers)}")
    else:
        # Manual entry of registration numbers and dates of birth
        col1, col2 = st.columns(2)
        
        with col1:
            reg_input = st.text_area("Enter Register Numbers (one per line)")
            if reg_input:
                reg_numbers = [reg.strip() for reg in reg_input.split("\n") if reg.strip()]
        
        with col2:
            dob_input = st.text_area("Enter Dates of Birth (one per line, format: DD-MM-YYYY)")
            if dob_input:
                dobs = [dob.strip() for dob in dob_input.split("\n") if dob.strip()]
        
        # Check if input lengths match
        if len(reg_numbers) != len(dobs) and reg_numbers and dobs:
            st.error("The number of Register Numbers and Dates of Birth must be the same.")
            reg_numbers = []
            dobs = []
        elif reg_numbers and dobs:
            # Display preview of the data
            preview_df = pd.DataFrame({
                "Register Number": reg_numbers,
                "Date of Birth": dobs
            })
            st.subheader("Data Preview")
            st.dataframe(preview_df)
            st.write(f"Total entries: {len(reg_numbers)}")
    
    # Browser and scraping settings
    st.subheader("Scraping Settings")
    browser_col1, browser_col2 = st.columns(2)
    
    with browser_col1:
        browser_type = st.selectbox("Browser", ["chrome", "firefox"], index=0)
        headless_mode = st.checkbox("Headless Mode", value=True, help="Run browser in background without UI")
    
    with browser_col2:
        delay_seconds = st.number_input("Delay Between Requests (seconds)", min_value=0, max_value=10, value=1)
        debug_mode = st.checkbox("Debug Mode", value=True, help="Show detailed debug information")
    
    # Scrape results button
    if st.button("Scrape Results") and reg_numbers and dobs:
        if len(reg_numbers) != len(dobs):
            st.error("The number of Register Numbers and Dates of Birth must be the same.")
        else:
            # Initialize list to store all results
            all_results = []
            
            # Create a progress bar
            total_students = len(reg_numbers)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Process each student
                for i, (reg_no, dob) in enumerate(zip(reg_numbers, dobs)):
                    # Update progress and status
                    progress_bar.progress((i + 1) / total_students)
                    status_text.text(f"Processing student {i+1} of {total_students}: {reg_no}")
                    
                    # Scrape result for this student
                    result, success = get_result(reg_no, dob, browser_type, headless_mode, delay_seconds, debug_mode)
                    all_results.append(result)
                    
                    # Update status with error information if scraping failed
                    if not success:
                        st.warning(f"Failed to process {reg_no}: {result.get('Error', 'Unknown error')}")
                
                # Update progress to 100% when done
                progress_bar.progress(1.0)
                status_text.text("Processing complete!")
                
                # Process the results for export
                st.subheader("Results Summary")
                st.session_state.df_successful, st.session_state.df_failed = process_results_for_export(all_results)
                
                if not st.session_state.df_successful.empty:
                    # Format the dataframe for better display
                    display_df = st.session_state.df_successful.copy()
                    
                    # Rename columns to make them more readable
                    column_renames = {}
                    for col in display_df.columns:
                        if '_' in col:
                            parts = col.split('_')
                            if len(parts) == 2:
                                subject_code = parts[0]
                                pos = int(parts[1])
                                
                                # Format based on position
                                if pos == 0:
                                    column_renames[col] = f"{subject_code} Mark1"
                                elif pos == 2:
                                    column_renames[col] = f"{subject_code} Mark2"
                                elif pos == 4:
                                    column_renames[col] = f"{subject_code} Mark3"
                                elif pos == 6:
                                    column_renames[col] = f"{subject_code} Result"
                    
                    # Apply renames if any
                    if column_renames:
                        display_df.rename(columns=column_renames, inplace=True)
                    
                    # Display the formatted dataframe
                    st.dataframe(display_df)
                    
                    # Show dimensions and summary
                    st.info(f"Output has {display_df.shape[0]} rows and {display_df.shape[1]} columns")
                    
                    # Export to Excel
                    excel_data = io.BytesIO()
                    with pd.ExcelWriter(excel_data, engine="openpyxl") as writer:
                        st.session_state.df_successful.to_excel(writer, sheet_name="Successful Results", index=False)
                        if not st.session_state.df_failed.empty:
                            st.session_state.df_failed.to_excel(writer, sheet_name="Failed Results", index=False)
                    
                    excel_data.seek(0)
                    st.download_button(
                        label="Download Results as Excel",
                        data=excel_data,
                        file_name="student_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.error("No successful results to display.")
                    if not st.session_state.df_failed.empty:
                        st.write("Failed Results:")
                        st.dataframe(st.session_state.df_failed)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error(traceback.format_exc())

# Q&A tab
with tab2:
    st.header("Query Your Results")
    st.write("After scraping results, you can ask questions about the data here.")
    
    # Add debug information about session state
    st.write(f"**Debug Info:** Data available in session state: {not st.session_state.df_successful.empty}")
    if not st.session_state.df_successful.empty:
        st.write(f"DataFrame shape: {st.session_state.df_successful.shape}")
    
    # Add option to upload previously exported results
    st.subheader("Load Previously Exported Results")
    st.write("If you've already scraped and exported results, you can upload the Excel file here:")
    uploaded_results = st.file_uploader("Upload previous results", type=["xlsx"])
    
    if uploaded_results is not None:
        try:
            # Load the uploaded Excel file
            xls = pd.ExcelFile(uploaded_results)
            sheet_names = xls.sheet_names
            
            if "Successful Results" in sheet_names:
                st.session_state.df_successful = pd.read_excel(uploaded_results, sheet_name="Successful Results")
                st.success(f"Successfully loaded {len(st.session_state.df_successful)} records from the uploaded file.")
                
                if "Failed Results" in sheet_names:
                    st.session_state.df_failed = pd.read_excel(uploaded_results, sheet_name="Failed Results")
            else:
                # If sheet names don't match expected format, try to load the first sheet
                st.session_state.df_successful = pd.read_excel(uploaded_results)
                st.success(f"Successfully loaded {len(st.session_state.df_successful)} records from the uploaded file.")
        except Exception as e:
            st.error(f"Error loading the Excel file: {str(e)}")
    
    # Example questions
    st.info("Example questions you can ask:\n" +
            "- Which student has the highest marks?\n" +
            "- How many students passed all subjects?\n" +
            "- What is the average grade for a specific subject?")
    
    # Allow the user to enter an API key directly if not found in environment variables
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.warning("Groq API key not found in environment variables.")
        api_key = st.text_input("Enter your Groq API key", type="password")
        if api_key:
            st.success("API key entered successfully!")
    
    # Q&A functionality
    question = st.text_input("Ask a question about the result data")
    
    if question and api_key:
        # Check if we have result data in memory
        if not st.session_state.df_successful.empty:
            try:
                with st.spinner("Thinking..."):
                    # Get the dataframe data as a string
                    df_sample = st.session_state.df_successful.head(20)  # Limit to 20 rows for performance
                    df_string = df_sample.to_string()
                    
                    # Create a prompt template
                    prompt_template = """
                    You are an AI assistant helping to analyze student results data.
                    
                    Here is a sample of the data (limited to 20 rows for brevity):
                    {df_string}
                    
                    The user asks: {question}
                    
                    Please provide a detailed and accurate answer based on this data.
                    If the information is not available in the data, please state that clearly.
                    """
                    
                    prompt = PromptTemplate(
                        template=prompt_template,
                        input_variables=["df_string", "question"]
                    )
                    
                    # Create Groq chat object
                    llm = ChatGroq(
                        temperature=0, 
                        groq_api_key=api_key,
                        model_name="llama3-70b-8192"  # Use an appropriate Groq model
                    )
                    
                    # Create the chain
                    chain = LLMChain(llm=llm, prompt=prompt)
                    
                    # Process the question
                    response = chain.run({
                        "df_string": df_string,
                        "question": question
                    })
                    
                    # Display the answer
                    st.write("### Answer")
                    st.write(response)
                    
                    # Display data source info
                    st.write("### Data Source")
                    st.write("Answer based on the student results data sample:")
                    st.dataframe(df_sample)
            
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")
                st.error(traceback.format_exc())
        else:
            st.warning("No result data available. Please scrape some results first.")
    elif question and not api_key:
        st.error("Please provide a valid Groq API key to use the Q&A functionality.")

# Display instructions
with st.expander("Instructions"):
    st.markdown("""
    ## How to use this tool:
    
    1. Prepare an Excel (.xlsx) file with the following columns:
       - **Register Number**: The student's registration number at University of Madras
       - **Date of Birth**: The student's date of birth (in DD/MM/YYYY format)
    
    2. Upload the Excel file using the button above
    
    3. Adjust settings in the sidebar if needed:
       - Select browser type (Chrome or Firefox)
       - Choose whether to run in headless mode
       - Set delay between requests
    
    4. Click "Start Scraping Results" to begin the process
    
    5. Once completed, you can:
       - Download the results as an Excel file
       - Ask questions about the data (requires Groq API key)
       
    ## Sample Questions to Ask:
    
    - How many students passed all subjects?
    - What is the average grade for a specific subject?
    - Which college had the highest pass rate?
    - List all students who failed any subject
    - What are the top performing colleges?
    - What are the most challenging subjects based on failure rates?
    """)

# Add a sample Excel file for download
st.sidebar.header("Sample Template")
sample_data = {
    'Register Number': ['123456789', '987654321'],
    'Date of Birth': ['01/01/2000', '15/06/1999']
}
sample_df = pd.DataFrame(sample_data)
sample_filename = "sample_template.xlsx"
sample_df.to_excel(sample_filename, index=False)

with open(sample_filename, "rb") as file:
    st.sidebar.download_button(
        label="Download Sample Template",
        data=file,
        file_name=sample_filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
