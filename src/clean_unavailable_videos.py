import pandas as pd
import requests
import time
from urllib.parse import urlparse, parse_qs

def is_video_available(url):
    """Check if a YouTube video is available by making a request to the URL"""
    try:
        # Add a timeout to avoid hanging
        response = requests.get(url, timeout=10)
        
        # Check if the response is successful
        if response.status_code == 200:
            content = response.text.lower()
            
            # More comprehensive indicators that a video is unavailable
            unavailable_indicators = [
                'video unavailable',
                'this video is not available',
                'video is private',
                'this video is private',
                'video has been removed',
                'this video has been removed',
                'video is no longer available',
                'this video is no longer available',
                'video unavailable in your country',
                'this video is not available in your country',
                'this video is not available in your region',
                'video is not available'
            ]
            
            # Check for video player elements that should be present for available videos
            video_player_indicators = [
                'video-player',
                'player-container',
                'ytd-player',
                'html5-video-player',
                'video-stream',
                'ytp-',
                'ytd-watch-flexy'
            ]
            
            # If any unavailable indicators are found, the video is unavailable
            for indicator in unavailable_indicators:
                if indicator in content:
                    return False
            
            # Check if video player elements are present (indicating video is available)
            has_video_player = any(indicator in content for indicator in video_player_indicators)
            
            # Additional checks for available video indicators
            available_indicators = [
                'visualizaciones',  # View count in Spanish
                'views',           # View count in English
                'subscribers',     # Subscriber count
                'suscriptores',    # Subscriber count in Spanish
                'like',            # Like button
                'dislike',         # Dislike button
                'share',           # Share button
                'compartir',       # Share button in Spanish
                'download',        # Download button
                'descargar'        # Download button in Spanish
            ]
            
            has_available_indicators = any(indicator in content for indicator in available_indicators)
            
            # Check for unavailable indicators
            unavailable_indicators = [
                'video no está disponible',  # Video is not available (Spanish)
                'video is not available',    # Video is not available (English)
                'el vídeo no está disponible',  # The video is not available (Spanish)
                'este vídeo no está disponible'  # This video is not available (Spanish)
            ]
            
            has_unavailable_indicators = any(indicator in content for indicator in unavailable_indicators)
            
            # If unavailable indicators are found, the video is unavailable
            if has_unavailable_indicators:
                return False
            
            # If video player elements are present AND available indicators are found, the video is likely available
            if has_video_player and has_available_indicators:
                return True
            
            # If no video player elements are found, the video is likely unavailable
            if not has_video_player:
                return False
            
            # If we have video player but no available indicators, be conservative and mark as unavailable
            return False
            
            # If we get here, the video appears to be available
            return True
        else:
            return False
            
    except Exception as e:
        print(f"Error checking {url}: {e}")
        return False

def clean_videos_csv(input_file, output_file, check_last_n=None):
    """Remove unavailable videos from the CSV file"""
    print(f"Reading video list from {input_file}...")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    df = df.dropna(subset=['youtube_url'])
    
    # If check_last_n is specified, only check the last N videos
    if check_last_n:
        df = df.tail(check_last_n)
        print(f"Checking only the last {check_last_n} videos...")
    else:
        print(f"Checking all {len(df)} videos in the dataset...")
    
    print(f"Found {len(df)} videos to check...")
    
    # Lists to track results
    available_videos = []
    unavailable_videos = []
    
    # Check each video
    for index, row in df.iterrows():
        url = row['youtube_url']
        print(f"Checking {index + 1}/{len(df)}: {url}")
        
        if is_video_available(url):
            available_videos.append(row)
            print(f"  ✓ Available")
        else:
            unavailable_videos.append(row)
            print(f"  ✗ Unavailable")
        
        # Add a small delay to be respectful to YouTube's servers
        time.sleep(1)
    
    # Create new DataFrame with only available videos
    if available_videos:
        available_df = pd.DataFrame(available_videos)
        available_df.to_csv(output_file, index=False)
        print(f"\n✓ Saved {len(available_videos)} available videos to {output_file}")
    else:
        print(f"\n⚠️  No available videos found!")
    
    # Report results
    print(f"\nResults:")
    print(f"  - Available videos: {len(available_videos)}")
    print(f"  - Unavailable videos: {len(unavailable_videos)}")
    print(f"  - Total checked: {len(df)}")
    
    # Show some examples of unavailable videos
    if unavailable_videos:
        print(f"\nExamples of unavailable videos:")
        for i, video in enumerate(unavailable_videos[:5]):  # Show first 5
            print(f"  {i+1}. {video['youtube_url']}")
        if len(unavailable_videos) > 5:
            print(f"  ... and {len(unavailable_videos) - 5} more")

def test_specific_video(url):
    """Test the availability detection on a specific video"""
    print(f"Testing video: {url}")
    is_available = is_video_available(url)
    print(f"Result: {'Available' if is_available else 'Unavailable'}")
    return is_available

def main():
    """Main function to clean the video list"""
    input_file = "data/videos_to_test.csv"
    output_file = "data/videos_to_test_cleaned.csv"
    
    print("YouTube Video Availability Checker")
    print("=" * 40)
    
    # Test the known unavailable video first
    test_url = "https://www.youtube.com/watch?v=SFjr-Qaaaf4"
    print("Testing known unavailable video...")
    test_specific_video(test_url)
    print()
    
    # Check if input file exists
    try:
        clean_videos_csv(input_file, output_file, check_last_n=None)  # Process all videos
    except FileNotFoundError:
        print(f"Error: {input_file} not found!")
        print("Please make sure the file exists in the data/ directory.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 