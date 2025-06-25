# This file is for debugging OpenWebUI required environment variables
# It will print all environment variables at container startup

import os
print("\n--- ENVIRONMENT VARIABLES ---")
for k, v in os.environ.items():
    print(f"{k}={v}")
print("--- END ENVIRONMENT VARIABLES ---\n")
