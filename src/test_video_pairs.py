import time
import csv
import itertools
import gc  # Garbage collection
import os
import random
from datetime import datetime, timedelta
import signal
import sys
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException, TimeoutException, SessionNotCreatedException, StaleElementReferenceException, InvalidSessionIdException
from selenium.common.exceptions import TimeoutException
import psutil  # For system resource monitoring

# Global flag to track if we're stuck
stuck_flag = False

def signal_handler(signum, frame):
    global stuck_flag
    print(f"\n⚠️  Signal {signum} received - script may be stuck!")
    print(f"⚠️  Signal triggered at: {datetime.now().strftime('%H:%M:%S')}")
    stuck_flag = True

# Set up signal handlers for stuck detection
signal.signal(signal.SIGALRM, signal_handler)

def read_video_urls(csv_file):
    """Read video URLs from CSV file"""
    urls = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip():  # Skip empty rows
                urls.append(row[0].strip())
    return urls

def read_already_compared_pairs(backup_file):
    """Read already compared video pairs from backup CSV file"""
    compared_pairs = set()
    if not os.path.exists(backup_file):
        print(f"Backup file {backup_file} not found, will test all pairs")
        return compared_pairs
    
    print(f"Reading backup file: {os.path.abspath(backup_file)}")
    print(f"File size: {os.path.getsize(backup_file)} bytes")
    
    try:
        with open(backup_file, 'r') as f:
            reader = csv.DictReader(f)
            row_count = 0
            for row in reader:
                row_count += 1
                # Store both directions (url1,url2) and (url2,url1) since order doesn't matter
                url1 = row['url1'].strip()
                url2 = row['url2'].strip()
                compared_pairs.add((url1, url2))
                compared_pairs.add((url2, url1))  # Add reverse pair too
        
        print(f"Processed {row_count} rows from backup file")
        print(f"Found {len(compared_pairs)//2} already compared pairs in {backup_file}")
            
    except Exception as e:
        print(f"Error reading backup file: {e}")
        import traceback
        traceback.print_exc()
    
    return compared_pairs

def get_videos_with_both_results(backup_file):
    """Get videos that already have both Cover and Not Cover results, considering manual feedback"""
    videos_with_both = set()
    
    if not os.path.exists(backup_file):
        return videos_with_both
    
    try:
        # Track results for each video
        video_results = {}
        
        with open(backup_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                url1 = row['url1'].strip()
                url2 = row['url2'].strip()
                result = row['result'].strip()
                feedback = row.get('feedback', '').strip()
                
                # Extract video IDs
                video1_id = url1.split('v=')[1] if 'v=' in url1 else url1
                video2_id = url2.split('v=')[1] if 'v=' in url2 else url2
                
                # Determine the effective result considering manual feedback
                effective_result = result
                if feedback:
                    if feedback == 'ok':
                        # User confirmed the automatic result is correct
                        effective_result = result
                    elif feedback == 'not-ok':
                        # User said the automatic result is wrong, so flip it
                        effective_result = 'Not Cover' if result == 'Cover' else 'Cover'
                    # If feedback is empty or other value, use the automatic result
                
                # Track results for each video
                for video_id in [video1_id, video2_id]:
                    if video_id not in video_results:
                        video_results[video_id] = set()
                    if effective_result:  # Only add if we have a valid result
                        video_results[video_id].add(effective_result)
        
        # Find videos with both Cover and Not Cover results
        for video_id, results in video_results.items():
            if 'Cover' in results and 'Not Cover' in results:
                videos_with_both.add(video_id)
        
        print(f"Found {len(videos_with_both)} videos with both Cover and Not Cover results (considering manual feedback)")
        
    except Exception as e:
        print(f"Error analyzing backup file: {e}")
    
    return videos_with_both

def find_videos_not_in_current_list(backup_file, current_urls):
    """Find videos in backup file that are not in the current test list"""
    backup_videos = set()
    backup_video_urls = {}  # Map video_id to full URL
    current_video_ids = set()
    
    # Get current video IDs
    for url in current_urls:
        video_id = url.split('v=')[1] if 'v=' in url else url
        current_video_ids.add(video_id)
    
    # Get all videos from backup file
    if os.path.exists(backup_file):
        try:
            with open(backup_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    url1 = row['url1'].strip()
                    url2 = row['url2'].strip()
                    
                    # Extract video IDs
                    video1_id = url1.split('v=')[1] if 'v=' in url1 else url1
                    video2_id = url2.split('v=')[1] if 'v=' in url2 else url2
                    
                    backup_videos.add(video1_id)
                    backup_videos.add(video2_id)
                    
                    # Store full URLs
                    backup_video_urls[video1_id] = url1
                    backup_video_urls[video2_id] = url2
        except Exception as e:
            print(f"Error reading backup file: {e}")
    
    # Find videos in backup but not in current list
    videos_not_in_current = backup_videos - current_video_ids
    
    print(f"\n{'='*60}")
    print("VIDEOS IN BACKUP BUT NOT IN CURRENT TEST LIST:")
    print(f"{'='*60}")
    print(f"Total videos in backup: {len(backup_videos)}")
    print(f"Total videos in current test list: {len(current_video_ids)}")
    print(f"Videos in backup but not in current list: {len(videos_not_in_current)}")
    
    if videos_not_in_current:
        print(f"\nVideos not in current test list:")
        for i, video_id in enumerate(sorted(videos_not_in_current), 1):
            full_url = backup_video_urls.get(video_id, f"https://www.youtube.com/watch?v={video_id}")
            print(f"  {i:2d}. {full_url}")
    
    print(f"{'='*60}")
    
    return videos_not_in_current

def generate_video_pairs(urls, already_compared_pairs, videos_with_both_results):
    """Generate pairs to ensure each video gets both Cover and Not Cover results"""
    pairs = []
    
    # Filter out videos that already have both Cover and Not Cover results
    filtered_urls = []
    skipped_videos = []
    
    for url in urls:
        video_id = url.split('v=')[1] if 'v=' in url else url
        if video_id in videos_with_both_results:
            skipped_videos.append(url)
        else:
            filtered_urls.append(url)
    
    print(f"Filtered out {len(skipped_videos)} videos that already have both Cover and Not Cover results")
    print(f"Remaining videos to test: {len(filtered_urls)}")
    
    # Print the remaining videos
    print(f"\nRemaining videos to test:")
    for i, url in enumerate(filtered_urls, 1):
        print(f"  {i:2d}. {url}")
    print()
    
    # Debug: check for duplicates in filtered_urls
    unique_urls = set(filtered_urls)
    if len(unique_urls) != len(filtered_urls):
        print(f"WARNING: Found {len(filtered_urls) - len(unique_urls)} duplicate URLs in filtered_urls!")
        duplicates = [url for url in filtered_urls if filtered_urls.count(url) > 1]
        print(f"Duplicate URLs: {duplicates}")
    else:
        print(f"✓ No duplicate URLs found in {len(filtered_urls)} remaining videos")
    print()
    
    # Track how many pairs each remaining video gets
    video_pair_count = {}
    for url in filtered_urls:
        video_id = url.split('v=')[1] if 'v=' in url else url
        video_pair_count[video_id] = 0
    
    # First, create pairs between the remaining videos themselves
    if len(filtered_urls) >= 2:
        print(f"Creating pairs between {len(filtered_urls)} remaining videos...")
        for i in range(len(filtered_urls)):
            for j in range(i + 1, len(filtered_urls)):
                url1 = filtered_urls[i]
                url2 = filtered_urls[j]
                
                # Explicit check to prevent self-pairing
                if url1 == url2:
                    print(f"WARNING: Skipping self-pair {url1} vs {url2}")
                    continue
                
                # Check if this pair has already been processed
                pair_exists = (url1, url2) in already_compared_pairs or (url2, url1) in already_compared_pairs
                
                if not pair_exists:
                    pair = (url1, url2)
                    pairs.append(pair)
                    video_pair_count[url1.split('v=')[1] if 'v=' in url1 else url1] += 1
                    video_pair_count[url2.split('v=')[1] if 'v=' in url2 else url2] += 1
                    print(f"Added pair between remaining videos: {url1} vs {url2}")
                else:
                    print(f"Skipping already processed pair: {url1} vs {url2}")
    
    # Then, create additional pairs with fully tested videos to ensure variety
    for current_url in filtered_urls:
        current_id = current_url.split('v=')[1] if 'v=' in current_url else current_url
        
        # Create up to 2 pairs per video (to ensure both Cover and Not Cover results)
        pairs_needed = 2 - video_pair_count[current_id]
        
        if pairs_needed > 0:
            # Get list of videos that already have both Cover and Not Cover results
            fully_tested_videos = []
            for url in urls:
                video_id = url.split('v=')[1] if 'v=' in url else url
                if video_id in videos_with_both_results:
                    fully_tested_videos.append(url)
            
            # Shuffle the list to get diversity
            random.shuffle(fully_tested_videos)
            
            # Track which fully tested videos we've already used for this current video
            used_partners = set()
            
            # Find videos that already have both Cover and Not Cover results to pair with
            for url in fully_tested_videos:  # Use shuffled list for diversity
                other_id = url.split('v=')[1] if 'v=' in url else url
                
                if other_id == current_id:  # Skip self
                    continue
                    
                if other_id in videos_with_both_results:  # Only pair with videos that have both results
                    # Skip if we've already used this partner for this video
                    if other_id in used_partners:
                        continue
                        
                    # Check if this pair has already been processed
                    pair_exists = (current_url, url) in already_compared_pairs or (url, current_url) in already_compared_pairs
                    
                    if not pair_exists:
                        pair = (current_url, url)
                        pairs.append(pair)
                        video_pair_count[current_id] += 1
                        used_partners.add(other_id)  # Mark this partner as used
                        print(f"Added pair: {current_url} vs {url} (with video that has both results)")
                        pairs_needed -= 1
                        
                        if pairs_needed <= 0:
                            break  # We have enough pairs for this video
    
    # Print summary
    print(f"\nGenerated {len(pairs)} pairs to test")
    
    return pairs

def calculate_estimated_completion_time(num_pairs, avg_time_per_test=30):
    """Calculate estimated completion time"""
    total_seconds = num_pairs * avg_time_per_test
    estimated_completion = datetime.now() + timedelta(seconds=total_seconds)
    return estimated_completion.strftime("%H:%M:%S"), total_seconds

def test_video_pair(driver, video1_url, video2_url, pair_index, total_pairs):
    """Test a single pair of videos"""
    global stuck_flag
    
    # Reset stuck flag for this test and cancel any existing alarm
    stuck_flag = False
    signal.alarm(0)  # Cancel any existing alarm
    
    # Reset refresh count for this test
    test_video_pair.refresh_count = 0
    
    print(f"\n{'='*60}")
    print(f"Testing pair {pair_index}/{total_pairs}")
    print(f"Video 1: {video1_url}")
    print(f"Video 2: {video2_url}")
    print(f"{'='*60}")
    
    try:
        # Navigate to the website
        driver.get("https://yt-coverhunter.fly.dev/")
        
        # Wait for the page to load with timeout
        wait = WebDriverWait(driver, 30)  # 30 second timeout
        
        # Check if the page loaded successfully
        title = driver.title
        print(f"Page title: {title}")
        
        # Look for the main elements
        try:
            # Check for the main heading
            heading = wait.until(EC.presence_of_element_located((By.TAG_NAME, "h1")))
            print(f"Main heading: {heading.text}")
            
            # Look for input fields
            input_fields = driver.find_elements(By.TAG_NAME, "input")
            print(f"Found {len(input_fields)} input fields")
            
            # Look for the compare button
            compare_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Compare videos') or contains(text(), 'Completed')]")
            print(f"Compare button found: {compare_button.text}")
            
            # Check if button already says "Completed!" (videos already compared)
            button_text = compare_button.text.strip()
            if "Completed" in button_text:
                print(f"Videos already compared! Button says: '{button_text}'")
                return True
            
            # Find the specific video URL inputs by ID
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
            
            # Check initial values
            initial_value1 = driver.execute_script("return arguments[0].value;", video1_input)
            initial_value2 = driver.execute_script("return arguments[0].value;", video2_input)
            print(f"Initial Input 1 value: '{initial_value1}'")
            print(f"Initial Input 2 value: '{initial_value2}'")
            
            # Enter test URLs with more careful approach
            test_url1 = video1_url
            test_url2 = video2_url
            
            # Clear and enter URL 1 with debugging
            print(f"\nEntering URL 1: {test_url1}")
            video1_input.clear()
            after_clear1 = driver.execute_script("return arguments[0].value;", video1_input)
            print(f"After clear - Input 1 value: '{after_clear1}'")
            
            # Use JavaScript to set the value and trigger input event
            driver.execute_script("""
                arguments[0].value = arguments[1];
                arguments[0].dispatchEvent(new Event('input', { bubbles: true, cancelable: true }));
            """, video1_input, test_url1)
            after_set1 = driver.execute_script("return arguments[0].value;", video1_input)
            print(f"After JavaScript set - Input 1 value: '{after_set1}'")
            
            # Clear and enter URL 2
            print(f"\nEntering URL 2: {test_url2}")
            video2_input.clear()
            after_clear2 = driver.execute_script("return arguments[0].value;", video2_input)
            print(f"After clear - Input 2 value: '{after_clear2}'")
            
            # Use JavaScript to set the value and trigger input event
            driver.execute_script("""
                arguments[0].value = arguments[1];
                arguments[0].dispatchEvent(new Event('input', { bubbles: true, cancelable: true }));
            """, video2_input, test_url2)
            after_set2 = driver.execute_script("return arguments[0].value;", video2_input)
            print(f"After JavaScript set - Input 2 value: '{after_set2}'")
            
            # Check values before clicking
            print(f"\nBefore clicking - Input 1 value: '{after_set1}'")
            print(f"Before clicking - Input 2 value: '{after_set2}'")
            
            # Click the compare button
            compare_button.click()
            print("Clicked compare button")
            print(f"Current URL: {driver.current_url}")
            
            # Wait for response with frequent checks
            print("Waiting for response...")
            max_wait_time = 120  # Maximum 2 minutes (120 seconds) for completion
            check_interval = 5   # Check every 5 seconds
            start_wait_time = time.time()
            last_check_time = start_wait_time
            
            # Set up a 180-second alarm for stuck detection (no progress) - ONLY NOW
            signal.alarm(180)
            
            while True:
                # Check if we're stuck globally
                if stuck_flag:
                    actual_waited = time.time() - start_wait_time
                    print(f"⚠️  Global stuck detection triggered after {actual_waited:.0f} seconds!")
                    signal.alarm(0)  # Cancel alarm
                    return "busy"
                
                current_time = time.time()
                waited_time = current_time - start_wait_time
                
                if waited_time >= max_wait_time:
                    print(f"Timeout after {waited_time:.0f} seconds (treating as busy)")
                    signal.alarm(0)  # Cancel alarm
                    return "busy"
                
                # Check if we're stuck (no progress for too long) - but limit refreshes
                if current_time - last_check_time > 30:  # If no check for 30 seconds, we're stuck
                    # Count how many times we've refreshed
                    refresh_count = getattr(test_video_pair, 'refresh_count', 0)
                    if refresh_count >= 3:  # Maximum 3 refreshes
                        print(f"Too many refreshes ({refresh_count}), treating as busy")
                        signal.alarm(0)  # Cancel alarm
                        return "busy"
                    
                    print(f"Stuck for {current_time - last_check_time:.0f} seconds, refreshing page... (refresh {refresh_count + 1}/3)")
                    try:
                        driver.refresh()
                        time.sleep(3)
                        start_wait_time = time.time()  # Reset timer
                        last_check_time = time.time()
                        signal.alarm(180)  # Reset alarm
                        test_video_pair.refresh_count = refresh_count + 1
                        continue
                    except Exception as e:
                        print(f"Failed to refresh page: {e}")
                        signal.alarm(0)  # Cancel alarm
                        return "busy"
                
                # Sleep for the check interval
                time.sleep(check_interval)
                last_check_time = time.time()
                
                # Calculate actual waited time after sleep
                actual_waited = time.time() - start_wait_time
                print(f"Waited {actual_waited:.0f}s, checking for result...")
                
                # Check if page is still responsive
                try:
                    driver.current_url
                except Exception as e:
                    print(f"Page became unresponsive: {e}")
                    signal.alarm(0)  # Cancel alarm
                    return False
                
                try:
                    # Check for success messages
                    success_elements = driver.find_elements(By.XPATH, "//div[contains(text(), 'Complete') or contains(text(), 'complete')]")
                    if success_elements:
                        print("Success/Result messages found:")
                        for elem in success_elements:
                            print(f"  - {elem.text}")
                        return True
                    
                    # Check for error messages
                    error_elements = driver.find_elements(By.XPATH, "//div[contains(text(), 'error') or contains(text(), 'Error') or contains(text(), 'failed') or contains(text(), 'Failed')]")
                    if error_elements:
                        print("Error messages found:")
                        for elem in error_elements:
                            if elem.text.strip():  # Only print non-empty error messages
                                print(f"  - {elem.text}")
                        return False
                    
                    # Check for busy message (but only after waiting at least 30 seconds)
                    if actual_waited > 30:
                        busy_elements = driver.find_elements(By.XPATH, "//div[contains(text(), 'busy') or contains(text(), 'Busy')]")
                        if busy_elements:
                            print("System is busy. Please try again in a few minutes.")
                            return "busy"
                    
                    # Check for progress messages (like "Processing audio...")
                    progress_elements = driver.find_elements(By.XPATH, "//div[contains(text(), 'Processing') or contains(text(), 'Downloading') or contains(text(), 'Est.')]")
                    if progress_elements:
                        progress_text = progress_elements[0].text if progress_elements else ""
                        print(f"Progress: {progress_text}")
                    
                    # If we've been waiting for more than 120 seconds with no clear result, treat as busy
                    if actual_waited > 120:
                        print(f"No clear result after {actual_waited:.0f} seconds, treating as busy")
                        return "busy"
                        
                except Exception as e:
                    print(f"Error checking page elements: {e}")
                    # If we can't even check elements, the page might be broken
                    if actual_waited > 30:
                        print("Page seems broken, treating as busy")
                        signal.alarm(0)  # Cancel alarm
                        return "busy"
            
            print(f"Timeout after {max_wait_time} seconds (treating as busy)")
            signal.alarm(0)  # Cancel alarm
            return "busy"
            
        except Exception as e:
            print(f"Error interacting with page elements: {e}")
            return False
            
    except Exception as e:
        print(f"Error testing pair: {e}")
        return False

def clear_browser_data(driver):
    """Clear browser cache and cookies to reduce memory usage"""
    try:
        driver.delete_all_cookies()
        driver.execute_script("window.localStorage.clear();")
        driver.execute_script("window.sessionStorage.clear();")
        print("🧹 Browser cache and cookies cleared")
    except Exception as e:
        print(f"Could not clear browser data: {e}")

def check_system_resources():
    """Check if system resources are available"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_percent = psutil.virtual_memory().percent
    
    print(f"System resources - CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%")
    
    # If system is overloaded, wait a bit
    if cpu_percent > 90 or memory_percent > 90:
        print("⚠️  System overloaded, waiting 30 seconds...")
        time.sleep(30)
        return False
    return True

def check_memory_usage():
    """Check current memory usage and return True if it's getting high"""
    try:
        # Try multiple import paths for different environments
        try:
            from app.utils.memory_logger import log_detailed_memory
        except ImportError:
            try:
                from utils.memory_logger import log_detailed_memory
            except ImportError:
                # Fallback: create a simple memory logger function
                def log_detailed_memory():
                    process = psutil.Process()
                    vms_total = process.memory_info().vms / (1024 * 1024)
                    print(f"\n=== Total VMS: {vms_total:.2f}MB ===")
                    
                    system_memory = psutil.virtual_memory()
                    print(f"=== System Memory ===")
                    print(f"Total: {system_memory.total / (1024**3):.2f}GB")
                    print(f"Available: {system_memory.available / (1024**3):.2f}GB")
                    print(f"Used: {system_memory.used / (1024**3):.2f}GB")
                    print(f"Free: {system_memory.free / (1024**3):.2f}GB")
                    print(f"Percent used: {system_memory.percent:.1f}%")
        
        # Log detailed memory information
        log_detailed_memory()
        
        # Check Python process memory
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Check system memory
        system_memory = psutil.virtual_memory()
        free_gb = system_memory.free / (1024**3)
        
        print(f"Python process memory: {memory_mb:.1f} MB")
        print(f"System free memory: {free_gb:.2f} GB")
        
        # If memory usage is high, suggest restart
        if memory_mb > 500:  # 500 MB threshold
            print("⚠️  High Python memory usage detected")
            return True
            
        # If system memory is low, also suggest restart
        if free_gb < 1.0:  # Less than 1 GB free
            print("⚠️  Low system memory detected")
            return True
            
        return False
        
    except Exception as e:
        # Fallback if anything goes wrong
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"Python process memory: {memory_mb:.1f} MB")
        
        # If memory usage is high, suggest restart
        if memory_mb > 500:  # 500 MB threshold
            print("⚠️  High memory usage detected")
            return True
        return False

def test_all_video_pairs():
    """Test all distinct pairs of videos from the CSV file"""
    
    # Configuration for same-machine optimization
    BATCH_SIZE = 5  # Process only 5 tests at a time
    BATCH_DELAY = 60  # Wait 1 minute between batches
    
    # Read video URLs and already compared pairs
    csv_file = "data/videos_to_test.csv"
    backup_file = "/data/compared_videos.csv"
    
    # Debug: list all backup files
    print("Available backup files:")
    for file in os.listdir('.'):
        if 'backup' in file and 'compared' in file and file.endswith('.csv'):
            print(f"  {file}")
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found!")
        return
    
    urls = read_video_urls(csv_file)
    if not urls:
        print("No video URLs found in CSV file!")
        return
    
    print(f"Found {len(urls)} video URLs: {urls}")
    
    already_compared_pairs = read_already_compared_pairs(backup_file)
    print(f"Found {len(already_compared_pairs)} already compared pairs")
    
    videos_with_both_results = get_videos_with_both_results(backup_file)
    print(f"Found {len(videos_with_both_results)} videos with both Cover and Not Cover results")
    
    # Find videos in backup but not in current test list
    find_videos_not_in_current_list(backup_file, urls)
    
    # Generate distinct pairs
    pairs = generate_video_pairs(urls, already_compared_pairs, videos_with_both_results)
    print(f"Generated {len(pairs)} distinct pairs to test")
    
    if not pairs:
        print("No new pairs to test!")
        return
    
    # Calculate estimated completion time
    estimated_time, total_seconds = calculate_estimated_completion_time(len(pairs), 120)  # 2 minutes per test
    print(f"Estimated completion time: {estimated_time} UTC (assuming 2 minutes per test)")
    print(f"Total estimated duration: {total_seconds//60} minutes {total_seconds%60} seconds")
    
    # Initialize counters
    successful_tests = 0
    failed_tests = 0
    busy_tests = 0
    skipped_tests = 0
    start_time = time.time()
    
    # Initialize Chrome driver with container-friendly options
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-background-timer-throttling")
    chrome_options.add_argument("--disable-backgrounding-occluded-windows")
    chrome_options.add_argument("--disable-renderer-backgrounding")
    chrome_options.add_argument("--disable-features=TranslateUI")
    chrome_options.add_argument("--disable-ipc-flooding-protection")
    chrome_options.add_argument("--remote-debugging-port=0")
    chrome_options.add_argument("--disable-images")
    chrome_options.add_argument("--disable-plugins")
    chrome_options.add_argument("--disable-web-security")
    chrome_options.add_argument("--disable-features=VizDisplayCompositor")
    chrome_options.add_argument("--memory-pressure-off")
    chrome_options.add_argument("--max_old_space_size=256")  # Reduced memory usage
    chrome_options.add_argument("--single-process")
    chrome_options.add_argument("--disable-background-networking")
    chrome_options.add_argument("--disable-default-apps")
    chrome_options.add_argument("--disable-sync")
    chrome_options.add_argument("--disable-translate")
    chrome_options.add_argument("--hide-scrollbars")
    chrome_options.add_argument("--mute-audio")
    chrome_options.add_argument("--no-first-run")
    chrome_options.add_argument("--disable-logging")
    chrome_options.add_argument("--disable-software-rasterizer")
    chrome_options.add_argument("--disable-setuid-sandbox")
    chrome_options.add_argument("--disable-gpu-sandbox")
    
    # Try to initialize Chrome driver with retries
    max_retries = 3
    retry_delay = 10  # seconds
    
    for attempt in range(max_retries):
        try:
            driver = webdriver.Chrome(options=chrome_options)
            print("Chrome driver initialized successfully")
            break
        except (SessionNotCreatedException, InvalidSessionIdException) as e:
            print(f"Attempt {attempt + 1}/{max_retries}: Failed to create Chrome session: {e}")
            if attempt < max_retries - 1:
                print(f"Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print("All retry attempts failed. Trying with additional container-friendly options...")
                
                # Add more container-friendly options
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu-sandbox")
        chrome_options.add_argument("--disable-software-rasterizer")
        chrome_options.add_argument("--disable-background-timer-throttling")
        chrome_options.add_argument("--disable-backgrounding-occluded-windows")
        chrome_options.add_argument("--disable-renderer-backgrounding")
        chrome_options.add_argument("--disable-features=TranslateUI")
        chrome_options.add_argument("--disable-ipc-flooding-protection")
        chrome_options.add_argument("--remote-debugging-port=0")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-setuid-sandbox")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--disable-features=VizDisplayCompositor")
        chrome_options.add_argument("--memory-pressure-off")
        chrome_options.add_argument("--max_old_space_size=256")  # Reduce memory usage further
        chrome_options.add_argument("--single-process")
        chrome_options.add_argument("--disable-background-networking")
        chrome_options.add_argument("--disable-default-apps")
        chrome_options.add_argument("--disable-sync")
        chrome_options.add_argument("--disable-translate")
        chrome_options.add_argument("--hide-scrollbars")
        chrome_options.add_argument("--mute-audio")
        chrome_options.add_argument("--no-first-run")
        chrome_options.add_argument("--disable-logging")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-software-rasterizer")
        chrome_options.add_argument("--disable-background-timer-throttling")
        chrome_options.add_argument("--disable-renderer-backgrounding")
        chrome_options.add_argument("--disable-backgrounding-occluded-windows")
        chrome_options.add_argument("--disable-features=TranslateUI")
        chrome_options.add_argument("--disable-ipc-flooding-protection")
        chrome_options.add_argument("--remote-debugging-port=0")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-images")
        chrome_options.add_argument("--disable-plugins")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--disable-features=VizDisplayCompositor")
        chrome_options.add_argument("--memory-pressure-off")
        chrome_options.add_argument("--max_old_space_size=256")
        chrome_options.add_argument("--single-process")
        chrome_options.add_argument("--disable-background-networking")
        chrome_options.add_argument("--disable-default-apps")
        chrome_options.add_argument("--disable-sync")
        chrome_options.add_argument("--disable-translate")
        chrome_options.add_argument("--hide-scrollbars")
        chrome_options.add_argument("--mute-audio")
        chrome_options.add_argument("--no-first-run")
        chrome_options.add_argument("--disable-logging")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-software-rasterizer")
        chrome_options.add_argument("--disable-background-timer-throttling")
        chrome_options.add_argument("--disable-renderer-backgrounding")
        chrome_options.add_argument("--disable-backgrounding-occluded-windows")
        chrome_options.add_argument("--disable-features=TranslateUI")
        chrome_options.add_argument("--disable-ipc-flooding-protection")
        chrome_options.add_argument("--remote-debugging-port=0")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--headless")
        
        try:
            driver = webdriver.Chrome(options=chrome_options)
            print("Chrome driver initialized with container-friendly options")
        except Exception as e:
            print(f"Failed to initialize Chrome driver with container options: {e}")
            print("Trying one more time with minimal options...")
            
            # Try with minimal options
            minimal_options = Options()
            minimal_options.add_argument("--headless")
            minimal_options.add_argument("--no-sandbox")
            minimal_options.add_argument("--disable-dev-shm-usage")
            minimal_options.add_argument("--disable-gpu")
            minimal_options.add_argument("--single-process")
            minimal_options.add_argument("--max_old_space_size=128")
            
            try:
                driver = webdriver.Chrome(options=minimal_options)
                print("Chrome driver initialized with minimal options")
            except Exception as final_e:
                print(f"Failed to initialize Chrome driver with minimal options: {final_e}")
                print("Please check if Chrome is properly installed and try again.")
                return
    
    # Test server health first
    print("Testing server health...")
    try:
        # Set a 30-second timeout for the health check
        driver.set_page_load_timeout(30)
        driver.get("https://yt-coverhunter.fly.dev/")
        time.sleep(5)
        
        # Try to find the compare button to see if the page loads properly
        try:
            compare_button = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//button[contains(text(), 'Compare')]"))
            )
            print("✓ Server appears to be working")
        except TimeoutException:
            print("⚠️  Server health check failed: Could not find compare button within 10 seconds")
            print("The server might be down or overloaded. Consider trying again later.")
            driver.quit()
            return
    
    except Exception as e:
        print(f"⚠️  Server health check failed: {e}")
        print("The server might be down or overloaded. Consider trying again later.")
        driver.quit()
        return
    
    try:
        for i, (url1, url2) in enumerate(pairs, 1):
            print(f"\n{'='*60}")
            print(f"Testing pair {i}/{len(pairs)}")
            print(f"Video 1: {url1}")
            print(f"Video 2: {url2}")
            print(f"{'='*60}")
            
            try:
                result = test_video_pair(driver, url1, url2, i, len(pairs))
                
                if result == "busy":
                    busy_tests += 1
                    print(f"⚠ Pair {i} is busy, moving to next pair...")
                elif result == True:
                    successful_tests += 1
                    print(f"✓ Pair {i} completed successfully")
                elif result == False:
                    failed_tests += 1
                    print(f"✗ Pair {i} failed")
                else:
                    skipped_tests += 1
                    print(f"⏭ Pair {i} skipped")
                    
            except InvalidSessionIdException as e:
                print(f"❌ Invalid session ID error: {e}")
                print("Session deleted as the browser has closed the connection")
                print("Waiting 1 minute and retrying...")
                time.sleep(60)  # Wait 1 minute as suggested
                
                # Restart the driver
                try:
                    driver.quit()
                except:
                    pass
                driver = webdriver.Chrome(options=chrome_options)
                print("Chrome driver restarted successfully")
                
                # Retry the same pair
                print(f"Retrying pair {i}...")
                try:
                    result = test_video_pair(driver, url1, url2, i, len(pairs))
                    
                    if result == "busy":
                        busy_tests += 1
                        print(f"⚠ Pair {i} is busy, moving to next pair...")
                    elif result == True:
                        successful_tests += 1
                        print(f"✓ Pair {i} completed successfully")
                    elif result == False:
                        failed_tests += 1
                        print(f"✗ Pair {i} failed")
                    else:
                        skipped_tests += 1
                        print(f"⏭ Pair {i} skipped")
                except Exception as retry_e:
                    print(f"❌ Retry also failed: {retry_e}")
                    failed_tests += 1
                    
            except Exception as e:
                print(f"❌ Error testing pair {i}: {e}")
                failed_tests += 1
                print("Continuing with next pair...")
                
                # Check if driver is still working, restart if needed
                try:
                    driver.current_url
                except:
                    print("Driver seems to be broken, restarting...")
                    try:
                        driver.quit()
                    except:
                        pass
                    driver = webdriver.Chrome(options=chrome_options)
                    print("Driver restarted successfully")
            
            # Wait between tests to avoid overwhelming the server
            if i < len(pairs):
                print("Waiting 15 seconds before next test...")
                time.sleep(15)
            
            # Clean up memory after each test
            gc.collect()  # Force garbage collection
            
            # Clear browser data to reduce memory usage
            clear_browser_data(driver)
            
            # Check memory usage and restart driver if needed
            if check_memory_usage():
                print("🔄 Restarting Chrome driver to free memory...")
                try:
                    driver.quit()
                except:
                    pass
                driver = webdriver.Chrome(options=chrome_options)
                print("Chrome driver restarted successfully")
                gc.collect()  # Clean up after restart
            
            # Process in batches to reduce server load
            if i % BATCH_SIZE == 0 and i < len(pairs):
                print(f"\n{'='*60}")
                print(f"Completed batch {i//BATCH_SIZE}. Taking a {BATCH_DELAY}s break to reduce server load...")
                print(f"{'='*60}")
                time.sleep(BATCH_DELAY)
                # Extra memory cleanup between batches
                gc.collect()
                gc.collect()  # Double cleanup
        
        # Calculate final statistics
        total_time = time.time() - start_time
        total_minutes = int(total_time // 60)
        total_seconds = int(total_time % 60)
        avg_time_per_test = total_time / len(pairs) if len(pairs) > 0 else 0
        success_rate = (successful_tests / len(pairs)) * 100 if len(pairs) > 0 else 0
        
        print(f"\n{'='*60}")
        print("FINAL RESULTS:")
        print(f"Total pairs: {len(pairs)}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Busy: {busy_tests}")
        print(f"Skipped: {skipped_tests}")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Total time: {total_minutes} minutes {total_seconds} seconds")
        print(f"Average time per test: {avg_time_per_test:.1f} seconds")
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