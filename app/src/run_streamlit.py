#!/usr/bin/env python3
"""
Run the Streamlit chatbot app.

Usage:
    python run_streamlit.py
    
Or from the command line:
    streamlit run streamlit.py
"""

import os
import subprocess
import sys


def main():
    """Run the Streamlit app."""
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    streamlit_file = os.path.join(current_dir, "streamlit.py")
    
    # Check if streamlit.py exists
    if not os.path.exists(streamlit_file):
        print(f"Error: {streamlit_file} not found!")
        sys.exit(1)
    
    # Run streamlit
    try:
        print("Starting Streamlit chatbot...")
        print(f"Running: streamlit run {streamlit_file}")
        subprocess.run(["streamlit", "run", streamlit_file], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: Streamlit is not installed or not in PATH")
        print("Install it with: pip install streamlit")
        sys.exit(1)

if __name__ == "__main__":
    main()