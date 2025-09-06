import pandas as pd
import csv
import shutil
import sys
from datetime import datetime


def log_message(message):
    """Log message with timestamp to both stdout and stderr for visibility"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    print(log_line, flush=True)
    print(log_line, file=sys.stderr, flush=True)


def review_video_pairs():
    """Apply feedback to rows with empty feedback when both youtube_url are in videos_to_test.csv with not empty clique identifier"""
    
    log_message("=== STARTING review_video_pairs.py ===")
    
    counter = 0
    processed_rows = []
    
    # Load videos_to_test.csv
    log_message("Loading videos_to_test.csv...")
    try:
        videos_to_test = pd.read_csv("data/videos_to_test.csv")
        videos_to_test = videos_to_test.dropna(subset=['youtube_url'])
        videos_to_test_urls = videos_to_test['youtube_url'].tolist()
        log_message(f"Loaded {len(videos_to_test_urls)} URLs from videos_to_test.csv")
    except Exception as e:
        log_message(f"ERROR loading videos_to_test.csv: {e}")
        return

    # Read the original file
    log_message("Loading compared_videos.csv...")
    try:
        df = pd.read_csv("/data/compared_videos.csv", keep_default_na=False, na_values=[])
        log_message(f"Loaded {len(df)} rows from compared_videos.csv")
    except Exception as e:
        log_message(f"ERROR loading compared_videos.csv: {e}")
        return

    # Process each row
    log_message("Processing rows...")
    for index, row in df.iterrows():
        if row['feedback'] != 'ok' and row['feedback'] != 'not-ok':
            if row['url1'] in videos_to_test_urls and row['url2'] in videos_to_test_urls:   
                
                youtube_url1 = row['url1']
                youtube_url2 = row['url2']
                
                try:
                    video_to_test_1 = videos_to_test.loc[videos_to_test['youtube_url'] == youtube_url1]
                    clique1 = video_to_test_1['clique'].values[0]
                    video_to_test_2 = videos_to_test.loc[videos_to_test['youtube_url'] == youtube_url2]
                    clique2 = video_to_test_2['clique'].values[0]
                    
                    if pd.notna(clique1) and pd.notna(clique2):
                        counter += 1
                        log_message(f"Processing row {index}: clique1={clique1}, clique2={clique2}, result={row['result']}")
                        
                        # Apply feedback logic
                        if (clique1 == clique2) and (row['result'] == 'Cover'):
                            df.at[index, 'feedback'] = 'ok'
                            feedback = 'ok'
                        elif (clique1 != clique2) and (row['result'] == 'Not Cover'):
                            df.at[index, 'feedback'] = 'ok'
                            feedback = 'ok'
                        else:
                            df.at[index, 'feedback'] = 'not-ok'
                            feedback = 'not-ok'
                        
                        processed_rows.append({
                            'index': index,
                            'url1': youtube_url1,
                            'url2': youtube_url2,
                            'clique1': clique1,
                            'clique2': clique2,
                            'result': row['result'],
                            'feedback': feedback
                        })
                        
                except Exception as e:
                    log_message(f"ERROR processing row {index}: {e}")
                    continue
                        
    log_message(f"Processed {counter} rows")
    
    # Save the updated dataframe
    if counter > 0:
        log_message("Saving updated file...")
        try:
            # Use simple pandas to_csv - this is the most reliable approach
            df.to_csv("/data/compared_videos.csv", index=False)
            log_message("File saved successfully")
            
            # Log summary of processed rows
            for row_info in processed_rows:
                log_message(f"Row {row_info['index']}: {row_info['url1']} vs {row_info['url2']} -> {row_info['feedback']}")
                
        except Exception as e:
            log_message(f"ERROR saving file: {e}")
    else:
        log_message("No rows to process")


if __name__ == "__main__":
    review_video_pairs()
