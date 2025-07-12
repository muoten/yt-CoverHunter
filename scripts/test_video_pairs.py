import csv
import itertools
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os

def read_video_urls(csv_file):
    """Read video URLs from CSV file"""
    urls = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip():  # Skip empty rows
                urls.append(row[0].strip())
    return urls

def generate_video_pairs(urls):
    """Generate all distinct pairs of video URLs"""
    pairs = []
    for url1, url2 in itertools.combinations(urls, 2):
        pairs.append((url1, url2))
    return pairs

def test_video_pair(driver, video1_url, video2_url, pair_index, total_pairs):
    """Test a single pair of videos"""
    print(f"\n{'='*60}")
    print(f"Testing pair {pair_index}/{total_pairs}")
    print(f"Video 1: {video1_url}")
    print(f"Video 2: {video2_url}")
    print(f"{'='*60}")
    
    try:
        # Navigate to the website
        driver.get("https://yt-coverhunter.fly.dev/")
        
        # Wait for the page to load
        wait = WebDriverWait(driver, 10)
        
        # Find the input fields
        try:
            video1_input = driver.find_element(By.ID, "video1")
            video2_input = driver.find_element(By.ID, "video2")
            print("Found input fields by ID")
        except:
            # Fallback to finding by CSS selector
            video_inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='text']")
            if len(video_inputs) >= 2:
                video1_input = video_inputs[0]
                video2_input = video_inputs[1]
                print("Found input fields by CSS selector")
            else:
                print("Could not find video URL input fields")
                return False
        
        # Find the compare button
        compare_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Compare videos') or contains(text(), 'Completed')]")
        
        # Check if button already says "Completed!" (videos already compared)
        button_text = compare_button.text.strip()
        if "Completed" in button_text:
            print(f"Videos already compared! Button says: '{button_text}'")
            return True
        
        # Clear and enter URL 1
        print(f"Entering URL 1: {video1_url}")
        video1_input.clear()
        time.sleep(0.5)
        
        # Use JavaScript to set the value and trigger input event
        driver.execute_script("""
            arguments[0].value = arguments[1];
            arguments[0].dispatchEvent(new Event('input', { bubbles: true, cancelable: true }));
        """, video1_input, video1_url)
        print(f"Input 1 value: '{video1_input.get_attribute('value')}'")
        
        # Clear and enter URL 2
        print(f"Entering URL 2: {video2_url}")
        video2_input.clear()
        time.sleep(0.5)
        
        # Use JavaScript to set the value and trigger input event
        driver.execute_script("""
            arguments[0].value = arguments[1];
            arguments[0].dispatchEvent(new Event('input', { bubbles: true, cancelable: true }));
        """, video2_input, video2_url)
        print(f"Input 2 value: '{video2_input.get_attribute('value')}'")
        
        # Check if URLs are the same (shouldn't happen with our pairs, but good to check)
        if video1_input.get_attribute('value') == video2_input.get_attribute('value'):
            print("ERROR: Both inputs have the same value!")
            return False
        
        # Click the compare button
        compare_button.click()
        print("Clicked compare button")
        
        # Wait for response with frequent checks
        print("Waiting for response...")
        max_wait_time = 20  # Maximum 20 iterations
        check_interval = 2   # Check every 2 seconds
        waited_time = 0
        
        while waited_time < max_wait_time:
            time.sleep(check_interval)
            waited_time += check_interval
            print(f"Waited {waited_time}s, checking for result...")
            
            try:
                # Look for any error messages (only non-empty ones)
                error_elements = driver.find_elements(By.XPATH, "//*[contains(text(), 'error') or contains(text(), 'Error') or contains(text(), 'failed') or contains(text(), 'Failed')]")
                actual_errors = []
                for error in error_elements:
                    if error.text.strip():  # Only count non-empty error messages
                        actual_errors.append(error.text.strip())
                
                if actual_errors:
                    print("Error messages found:")
                    for error in actual_errors:
                        print(f"  - {error}")
                    return False
                
                # Look for success/result messages
                try:
                    progress_text = driver.find_element(By.CSS_SELECTOR, "div#progress-text.progress-status")
                    print(progress_text.text.strip())
                    if progress_text.text.strip() and "complete" in progress_text.text.lower():
                        print(f"Success/Result messages found: {progress_text.text.strip()}")
                        return True
                except:
                    # Fallback to general search
                    success_elements = driver.find_elements(By.XPATH, "//*[contains(text(), 'complete') or contains(text(), 'Complete') or contains(text(), 'complete!') or contains(text(), 'Complete!')]")
                    actual_successes = []
                    for success in success_elements:
                        if success.text.strip():  # Only count non-empty success messages
                            actual_successes.append(success.text.strip())
                    
                    if actual_successes:
                        print("Success/Result messages found:")
                        for success in actual_successes:
                            print(f"  - {success}")
                        return True
                
                # Look for loading indicators
                loading_elements = driver.find_elements(By.XPATH, "//*[contains(text(), 'loading') or contains(text(), 'processing') or contains(text(), 'Loading')]")
                actual_loadings = []
                for loading in loading_elements:
                    if loading.text.strip():  # Only count non-empty loading messages
                        actual_loadings.append(loading.text.strip())
                
                if actual_loadings:
                    print("Still loading...")
                    print("Loading messages:")
                    for loading in actual_loadings:
                        print(f"  - {loading}")
                    continue  # Keep waiting
                
                # Check for any text that might indicate completion
                all_text_elements = driver.find_elements(By.XPATH, "//*[text()]")
                relevant_texts = []
                for elem in all_text_elements:
                    text = elem.text.strip()
                    #print(text)
                    if text and any(keyword in text.lower() for keyword in ['complete', 'complete!']):
                        relevant_texts.append(text)
                
                if relevant_texts:
                    print("Relevant text found:")
                    for text in relevant_texts:
                        print(f"  - {text}")
                    return True
                
                # If no clear indicators, continue waiting
                print("No clear result yet, continuing to wait...")
                
            except Exception as e:
                print(f"Error checking for responses: {e}")
                continue
        
        print(f"Timeout after {max_wait_time} seconds")
        return False
            
    except Exception as e:
        print(f"Error testing pair: {e}")
        return False

def test_all_video_pairs():
    """Test all distinct pairs of videos from the CSV file"""
    
    # Read video URLs from CSV
    csv_file = "data/videos_to_test.csv"
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found!")
        return
    
    urls = read_video_urls(csv_file)
    print(f"Found {len(urls)} video URLs: {urls}")
    
    # Generate all distinct pairs
    pairs = generate_video_pairs(urls)
    print(f"Generated {len(pairs)} distinct pairs to test")
    
    if not pairs:
        print("No pairs to test!")
        return
    
    # Configure Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--start-maximized")
    # Uncomment the line below to run in headless mode
    # chrome_options.add_argument("--headless")
    
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        successful_tests = 0
        failed_tests = 0
        
        for i, (url1, url2) in enumerate(pairs, 1):
            print(f"\n{'='*60}")
            print(f"PROGRESS: {i}/{len(pairs)} pairs tested")
            print(f"Successful: {successful_tests}, Failed: {failed_tests}")
            print(f"{'='*60}")
            
            success = test_video_pair(driver, url1, url2, i, len(pairs))
            
            if success:
                successful_tests += 1
                print(f"✓ Pair {i} completed successfully")
            else:
                failed_tests += 1
                print(f"✗ Pair {i} failed")
            
            # Wait between tests to avoid overwhelming the server
            if i < len(pairs):
                print("Waiting 2 seconds before next test...")
                time.sleep(2)
        
        print(f"\n{'='*60}")
        print("FINAL RESULTS:")
        print(f"Total pairs tested: {len(pairs)}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success rate: {(successful_tests/len(pairs)*100):.1f}%")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Error in main test loop: {e}")
        
    finally:
        print("\nClosing browser...")
        driver.quit()

if __name__ == "__main__":
    print("Testing all video pairs from videos_to_test.csv...")
    test_all_video_pairs()
    print("Test completed!") 