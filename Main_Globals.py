import logging

# Custom formatter for INFO logs to align the type field (makes it easy to read)
class CustomFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno == logging.INFO:
            # Check if the message follows the expected format (ticker - type - message)
            msg_str = str(record.msg)  # Convert to string in case it's an f-string or other object
            parts = msg_str.split(' - ', 2) if ' - ' in msg_str else [msg_str]
            
            if len(parts) >= 3:
                ticker = parts[0]
                type_field = parts[1]
                message = parts[2]
                
                # Format with type field padded to 12 characters
                record.msg = f"{ticker} - {type_field:<15} - {message}"
            
        return super().format(record)
    

def SetupLogging():
    """
    Sets up the logging system with a custom formatter that aligns the 'type' field
    to 12 characters for better readability in log files.
    """
    global logger
    
    # Create the logger
    logger = logging.getLogger('watchlist_extractor')
    
    # Configure basic logging
    logging.basicConfig(
        filename='Logs.txt',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Apply custom formatter to root logger handlers
    for handler in logging.getLogger().handlers:
        handler.setFormatter(CustomFormatter('%(asctime)s - %(levelname)s - %(message)s', '%H:%M:%S'))
    
    # Ensure our specific logger inherits from root
    logger.propagate = True

SetupLogging() # Initialize logger


def ErrorHandler(fileName, function, message, lineNumber):
    logger.error(f"{fileName} - {function}() - Line {lineNumber}: {message}")
    print(f"\nERROR: {fileName} - {function}() - Line {lineNumber}: {message}*********\n")
