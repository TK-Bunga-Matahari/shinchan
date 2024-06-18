# import library
import os
from dotenv import load_dotenv

try:
    # Load environment variables from .env file
    load_dotenv()

    # Get License Information from Environment Variables
    wls_access_id = os.getenv("WLSACCESSID")
    wls_secret = os.getenv("WLSSECRET")
    license_id = os.getenv("LICENSEID")

    if wls_access_id is None or wls_secret is None or license_id is None:
        license_params = {}
    else:
        license_params = {
            "WLSACCESSID": wls_access_id,
            "WLSSECRET": wls_secret,
            "LICENSEID": int(license_id),
        }
except Exception as e:
    license_params = {}
    print(f"Error in get the Gurobi License: {e}")
