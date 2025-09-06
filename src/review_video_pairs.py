import pandas as pd
import csv
import shutil


def review_video_pairs():
    " print the rows with empty feedback when both youtube_url are in videos_to_test.csv with not empty clique identifier"
    counter = 0
    # if column feedback is empty and both youtube_url are in videos_to_test.csv with not empty clique identifier, mark as 'ok' if same clique, not-ok if different
    videos_to_test = pd.read_csv("data/videos_to_test.csv")
    videos_to_test = videos_to_test.dropna(subset=['youtube_url'])
    videos_to_test_urls = videos_to_test['youtube_url'].tolist()

    # Read the original file with exact same parameters
    df = pd.read_csv("/data/compared_videos.csv", keep_default_na=False, na_values=[])
    for index, row in df.iterrows():

        if row['feedback'] != 'ok' and row['feedback'] != 'not-ok':
            #print(row)
            if row['url1'] in videos_to_test_urls and row['url2'] in videos_to_test_urls:   
                
                youtube_url1 = row['url1']
                youtube_url2 = row['url2']
                video_to_test_1 = videos_to_test.loc[videos_to_test['youtube_url'] == youtube_url1]
                clique1 = video_to_test_1['clique'].values[0]
                video_to_test_2 = videos_to_test.loc[videos_to_test['youtube_url'] == youtube_url2]
                clique2 = video_to_test_2['clique'].values[0]
                
                if pd.notna(clique1) and pd.notna(clique2):
                    counter += 1
                    print("Applying feedback to row ", index, "clique1: ", clique1, "clique2: ", clique2, "result: ", row['result'])
                    # in this case we have to edit the feedback column in df
                    if (clique1 == clique2) and (row['result'] == 'Cover'):
                        df.at[index, 'feedback'] = 'ok'
                        
                    elif (clique1 != clique2) and (row['result'] == 'Not Cover'):
                        df.at[index, 'feedback'] = 'ok'
                        
                    else:
                        df.at[index, 'feedback'] = 'not-ok'
                        
    print("feedback applied to ", counter, " rows")
    
    # Copy original file to preserve exact format
    shutil.copy("/data/compared_videos.csv", "/data/compared_videos_reviewed.csv")
    
    # Now update only the specific lines that changed
    with open("/data/compared_videos_reviewed.csv", 'rb') as f:
        content = f.read()
        lines = content.split(b'\r\n')
    
    # Update the specific lines that were modified
    for index, row in df.iterrows():
        if index < len(lines) - 1:  # Skip header line
            # Split the line and update feedback column (5th column, index 4)
            line_str = lines[index + 1].decode('utf-8')
            parts = line_str.split(',')
            if len(parts) >= 5:
                parts[4] = str(row['feedback'])  # Update feedback column
                lines[index + 1] = ','.join(parts).encode('utf-8')
    
    # Write back the file with exact same format (preserve Windows line endings)
    with open("/data/compared_videos_reviewed.csv", 'wb') as f:
        f.write(b'\r\n'.join(lines))              
    shutil.copy("/data/compared_videos_reviewed.csv", "/data/compared_videos.csv")


if __name__ == "__main__":
    review_video_pairs()