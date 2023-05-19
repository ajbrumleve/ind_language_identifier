from functools import wraps
from datetime import datetime

def log_to_file(filename):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            changes = input("What changes were made before this run?")
            # Write the classification report to a file
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            report_with_datetime = f"\n\nTimestamp: {timestamp}\n{changes}\n\nClassification Report:\n{result}"

            # Write the report to a file
            with open(filename, 'a') as file:
                file.write(report_with_datetime)

            return result

        return wrapper

    return decorator