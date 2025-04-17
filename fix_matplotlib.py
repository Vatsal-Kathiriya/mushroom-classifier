# Fix matplotlib backend issues in Flask
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
print("Set matplotlib backend to 'Agg' for thread safety")