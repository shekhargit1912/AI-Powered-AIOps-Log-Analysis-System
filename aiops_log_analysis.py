"""
My AIOps Log Analysis System
============================

Started this project to solve memory leak detection issues at work.
Traditional monitoring tools kept missing critical problems in INFO logs.

What it does:
- Automatically learns suspicious patterns from log data
- Uses multiple ML algorithms to reduce false positives  
- Explains why each log entry was flagged as anomalous
- Saves results in easy-to-review CSV files

Built over several weeks of experimentation with different approaches.
The TF-IDF + ensemble method combo works best for my use cases.

Author: Shekhar Chaugule
Version: 2.1 (much better than my first attempt!)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import re
import warnings
warnings.filterwarnings('ignore')  # Got tired of sklearn warnings during development

# My log analysis project - evolved from simple anomaly detection
# TODO: Maybe add real-time processing later
# FIXME: Need better error handling for malformed logs
# NOTE: Current version works well for batch analysis

log_file_path = r"C:\Users\ShekharChaugule\Documents\AIOPSLOG\sample_logs.txt"
print(f"Loading logs from: {log_file_path}")

try:
    with open(log_file_path, "r") as file:
        logs = file.readlines()
    print(f"Successfully loaded {len(logs)} log entries")
except FileNotFoundError:
    print("Error: Log file not found!")
    exit(1)

# my custom log parser - took a while to get this right
def parse_my_logs(log_lines):
    """Parse logs the way I need them - handles different formats I've seen"""
    parsed_data = []
    failed_count = 0
    
    for line_num, log in enumerate(log_lines):
        try:
            parts = log.strip().split(" ", 3)
            if len(parts) < 4:
                failed_count += 1
                continue  # Skip weird lines
            
            # combine date and time - this format works for my logs
            timestamp = parts[0] + " " + parts[1]
            level = parts[2]
            message = parts[3]
            parsed_data.append([timestamp, level, message])
            
        except Exception as e:
            # print(f"Debug: Failed to parse line {line_num}: {e}")  # Used for debugging
            failed_count += 1
            continue
    
    if failed_count > 0:
        print(f"Warning: Couldn't parse {failed_count} lines (probably malformed)")
    
    return parsed_data

# Parse all the logs
print("Parsing log entries...")
data = parse_my_logs(logs)
df = pd.DataFrame(data, columns=["timestamp", "level", "message"])
print(f"Successfully parsed {len(df)} log entries")

# Convert timestamps - pandas is pretty good at this
df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')

# My scoring system for log levels - learned this from experience
# INFO is usually fine, CRITICAL means wake me up at 3am!
my_level_scores = {"INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}
df["level_score"] = df["level"].map(my_level_scores)

# Message length might be important - longer messages often mean trouble
df["msg_len"] = df["message"].apply(len)

# quick check of what we're working with
print(f"Log levels found: {df['level'].value_counts().to_dict()}")
print(f"Average message length: {df['msg_len'].mean():.1f} characters")

# my anomaly detection system - this took me weeks to get right!
def detect_log_anomalies(log_df):
    """
    My approach to finding weird stuff in logs
    Tried different methods, this combo works best for my use case
    """
    
    print("Analyzing log patterns...")
    
    # Step 1: Learn what words are important using TF-IDF
    # Had to experiment with these parameters - 85 features works better than 100 for my logs
    my_vectorizer = TfidfVectorizer(
        max_features=85,  # Found this works better than default
        stop_words='english', 
        ngram_range=(1, 2)  # Single words and pairs
    )
    
    text_matrix = my_vectorizer.fit_transform(log_df['message']).toarray()
    
    # step 2: Combine text features with numeric ones
    numeric_data = log_df[["level_score", "msg_len"]].values
    
    # Scale the numbers so they play nice with text features
    my_scaler = StandardScaler()
    scaled_numbers = my_scaler.fit_transform(numeric_data)
    
    # combine everything into one big feature matrix
    combined_features = np.hstack([scaled_numbers, text_matrix])
    
    print("Running anomaly detection algorithms...")
    
    # Method 1: Isolation Forest - good at finding outliers
    # tweaked contamination to 0.08 after testing with my data
    iso_detector = IsolationForest(
        contamination=0.08,  # My sweet spot after experimentation
        random_state=42,     # For reproducible results
        n_estimators=150     # Reduced from 200, faster and works just as well
    )
    iso_results = iso_detector.fit_predict(combined_features)
    
    # Method 2: DBSCAN clustering - finds weird clusters
    # eps=0.6 works better for my log data than 0.5
    cluster_detector = DBSCAN(eps=0.6, min_samples=4)  # min_samples=4 reduces noise
    cluster_labels = cluster_detector.fit_predict(combined_features)
    cluster_results = np.where(cluster_labels == -1, -1, 1)
    
    # Method 3: Statistical outlier detection
    # Using Z-score > 2.3 instead of 2.5 - catches more edge cases
    z_scores = np.abs((scaled_numbers - scaled_numbers.mean(axis=0)) / scaled_numbers.std(axis=0))
    stat_results = np.where(np.max(z_scores, axis=1) > 2.3, -1, 1)
    
    # Combine all three methods - need at least 2 to agree
    votes = iso_results + cluster_results + stat_results
    final_decisions = np.where(votes <= -1, -1, 1)
    
    # Figure out what patterns the system learned
    feature_names = my_vectorizer.get_feature_names_out()
    importance_scores = np.abs(text_matrix).mean(axis=0)
    
    # Get top suspicious patterns
    top_indices = np.argsort(importance_scores)[-8:]  # Top 8 instead of 10
    suspicious_patterns = [feature_names[i] for i in top_indices]
    
    print(f"Found these suspicious patterns: {', '.join(suspicious_patterns)}")
    
    return final_decisions, suspicious_patterns, text_matrix, feature_names

# Run my anomaly detection system
df["anomaly_score"], suspicious_words, text_features, all_features = detect_log_anomalies(df)

# Convert scores to readable labels
df["status"] = df["anomaly_score"].apply(lambda x: "❌ Anomaly" if x == -1 else "✅ Normal")

# My reasoning system - explains why something looks suspicious
def explain_why_anomaly(log_row, row_idx):
    """
    Figure out why this log entry was flagged
    This helps me understand what the system is thinking
    """
    if log_row["status"] == "✅ Normal":
        return "Looks normal to me"
    
    explanations = []
    
    # Check what text patterns triggered it
    row_features = text_features[row_idx]
    top_feature_idx = np.argsort(row_features)[-4:]  # Top 4 patterns
    triggered_patterns = [all_features[i] for i in top_feature_idx if row_features[i] > 0]
    
    if triggered_patterns:
        explanations.append(f"Suspicious patterns found: {', '.join(triggered_patterns)}")
    
    # Cheeck if message length is weird
    avg_len = df["msg_len"].mean()
    std_len = df["msg_len"].std()
    if abs(log_row["msg_len"] - avg_len) > 2 * std_len:
        explanations.append("Message length is unusual")
    
    # High severity levels are always suspicious
    if log_row["level"] in ["ERROR", "CRITICAL"]:
        explanations.append(f"High severity: {log_row['level']}")
    
    # Check for overlap with known suspicious words
    msg_words = set(log_row["message"].lower().split())
    matching_suspicious = msg_words.intersection(set(suspicious_words))
    if matching_suspicious:
        explanations.append(f"Contains suspicious terms: {', '.join(matching_suspicious)}")
    
    # Look for performance metrics (usually indicate problems)
    if "%" in log_row["message"] and any(c.isdigit() for c in log_row["message"]):
        explanations.append("Contains performance metrics")
    
    # Fallback explanation
    if not explanations:
        explanations.append("Multiple detection methods flagged this")
    
    return "; ".join(explanations)

# Generate explanations for each log entry
print("Generating explanations for detected anomalies...")
df["explanation"] = [explain_why_anomaly(row, idx) for idx, row in df.iterrows()]

# Sort by time to see the sequence of events
df = df.sort_values("timestamp")

# Show me what we found
anomaly_logs = df[df["status"] == "❌ Anomaly"]
print(f"\n🔍 Found {len(anomaly_logs)} suspicious log entries:")
if len(anomaly_logs) > 0:
    # Show first few anomalies with key info
    display_cols = ['timestamp', 'level', 'message', 'explanation']
    print(anomaly_logs[display_cols].head(8).to_string(index=False))
else:
    print("No anomalies detected - system looks healthy!")

# My file saving functions - keep both detailed and summary reports
def save_full_analysis(data_df, filename="my_log_analysis.csv"):
    """Save everything - I like having all the data for later analysis"""
    try:
        data_df.to_csv(filename, index=False)
        
        total = len(data_df)
        anomalies = len(data_df[data_df["status"] == "❌ Anomaly"])
        normal = total - anomalies
        
        print(f"\n💾 Full analysis saved to '{filename}'")
        print(f"📊 Processed {total} log entries")
        print(f"✅ Normal: {normal} entries")
        print(f"❌ Suspicious: {anomalies} entries")
        
        if anomalies > 0:
            print(f"   Anomaly rate: {(anomalies/total)*100:.1f}%")
        
    except Exception as e:
        print(f"Error saving full report: {e}")

def save_just_anomalies(data_df, filename="suspicious_logs.csv"):
    """Save only the problematic entries - easier for quick review"""
    try:
        problem_logs = data_df[data_df["status"] == "❌ Anomaly"].copy()
        
        if len(problem_logs) > 0:
            # Keep the important columns for incident response
            important_cols = ['timestamp', 'level', 'message', 'status', 'explanation']
            problem_logs[important_cols].to_csv(filename, index=False)
            
            print(f"🚨 Suspicious entries saved to '{filename}'")
            print(f"   📋 {len(problem_logs)} entries need attention")
            
            # Quick summary of what we found
            level_counts = problem_logs['level'].value_counts()
            print(f"   Breakdown: {level_counts.to_dict()}")
        else:
            print("✅ No suspicious entries found - nothing to save")
            
    except Exception as e:
        print(f"Error saving anomaly report: {e}")

# Save both reports - I find both useful
print("\nSaving analysis results...")
save_full_analysis(df)
save_just_anomalies(df)

print("\n" + "="*50)
print("Analysis complete! 🎉")
print("Check the CSV files for detailed results.")
print("="*50)
