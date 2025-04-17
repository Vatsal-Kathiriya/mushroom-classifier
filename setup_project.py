import os

def create_directory_structure():
    """
    Create the necessary directory structure for the project
    """
    directories = [
        'data',
        'data/processed',
        'models',
        'models/evaluation',
        'notebooks',
        'src',
        'static/img',
        'static/css',
        'static/js',
        'templates'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

if __name__ == "__main__":
    create_directory_structure()
    print("Project structure setup complete!")