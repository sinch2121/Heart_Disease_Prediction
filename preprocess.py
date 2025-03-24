import numpy as np
import pandas as pd

def preprocess_input(input_data):
    try:
        # Convert to DataFrame for easier processing
        df = pd.DataFrame([input_data])

        # Perform necessary transformations
        # Example: Scaling or Encoding if needed
        # Assuming no further preprocessing is needed for now

        return df.values
    
    except Exception as e:
        raise ValueError(f"Error in preprocessing input: {str(e)}")
