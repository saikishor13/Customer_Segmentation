import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
def switch_to_register():
    tab_control.select(1)  # Switch to the register tab

def switch_to_login():
    tab_control.select(0)  # Switch to the login tab

def login():
    # Validate login credentials
    username = username_entry.get()
    password = password_entry.get()
    
    # Perform login validation logic here
    
    # If login successful, switch to browse tab
    tab_control.select(2)  # Switch to the browse tab
    login_window.state('zoomed')  # Maximize window

def register():

    
    # Switch to the login tab after successful registration
    switch_to_login()

def browse_file():
    # Open a file dialog for the user to select a CSV file
    filename = filedialog.askopenfilename(initialdir="/", title="Select a CSV file", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
    if filename:
        # Read the CSV file using pandas
        df = pd.read_csv(filename)
        process_data(df)
        
        # Switch to the "Browse CSV" tab in the GUI
        tab_control.select(2)  # Switch to the browse tab

def process_data(df):
# Data analysis steps
    df.head()
    df.info()
    df.shape
    df.describe().T
    df.isnull().sum()
    df.duplicated().value_counts()
    ax = sns.countplot(x='Gender', data=df)
    ax.bar_label(container=ax.containers[0], labels=df['Gender'].value_counts(ascending=True))
    plt.show()
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 3, 1)
    sns.boxplot(x=df.Gender, y=df.Age)
    plt.title('Distribution of Age')

    plt.subplot(1, 3, 2)
    sns.boxplot(x=df.Gender, y=df['Annual Income (k$)'])
    plt.title('Distribution of Income')

    plt.subplot(1, 3, 3)
    sns.boxplot(x=df.Gender, y=df['Spending Score (1-100)'])
    plt.title('Distribution of Spending')
    plt.show()
    female = df[df.Gender == 'Female']
    male = df[df.Gender == 'Male']

    plt.figure(figsize=(20, 5))
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df, x='Age', y='Annual Income (k$)', hue='Gender')
    plt.xlabel('Age')
    plt.ylabel('Income')

    plt.subplot(1, 2, 2)
    sns.scatterplot(data=df, x='Age', y='Spending Score (1-100)', hue='Gender')
    plt.xlabel('Age')
    plt.ylabel('Spending')
    plt.show()

    x = df.iloc[:, [3, 4]].values
    wcss = []

    for cluster in range(1, 11):
        kmeans = KMeans(n_clusters=cluster, init='k-means++', random_state=42)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)
    
    plt.plot(range(1, 11), wcss, 'o--')
    plt.title('Elbow Method')
    plt.xlabel('No of Clusters')
    plt.ylabel('WCSS')
    plt.show()

    kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
    y = kmeans.fit_predict(x)
    df['Cluster'] = kmeans.labels_

    plt.scatter(x[y == 0, 0], x[y == 0, 1], s=20, c='red', label='Cluster 1')
    plt.scatter(x[y == 1, 0], x[y == 1, 1], s=20, c='blue', label='Cluster 2')
    plt.scatter(x[y == 2, 0], x[y == 2, 1], s=20, c='green', label='Cluster 3')
    plt.scatter(x[y == 3, 0], x[y == 3, 1], s=20, c='cyan', label='Cluster 4')
    plt.scatter(x[y == 4, 0], x[y == 4, 1], s=20, c='magenta', label='Cluster 5')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='yellow', label='Centroids')
    plt.title('Clusters of Customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.show()

    oneData = df.iloc[y == 0, :]
    twoData = df.iloc[y == 1, :]
    threeData = df.iloc[y == 2, :]
    fourData = df.iloc[y == 3, :]
    fiveData = df.iloc[y == 4, :]

    ax = sns.barplot(x=['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5'],
                     y=[len(oneData), len(twoData), len(threeData), len(fourData), len(fiveData)])
    ax.bar_label(container=ax.containers[0], labels=[len(oneData), len(twoData), len(threeData), len(fourData), len(fiveData)])
    plt.show()

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    ax = sns.barplot(x=['Minimum Age', 'Maximum Age'], y=[oneData['Age'].min(), oneData['Age'].max()])
    ax.bar_label(container=ax.containers[0], labels=[oneData['Age'].min(), oneData['Age'].max()])
    plt.title('Cluster One')

    plt.subplot(2, 3, 2)
    ax = sns.barplot(x=['Minimum Age', 'Maximum Age'], y=[twoData['Age'].min(), twoData['Age'].max()])
    ax.bar_label(container=ax.containers[0], labels=[twoData['Age'].min(), twoData['Age'].max()])
    plt.title('Cluster Two')

    plt.subplot(2, 3, 3)
    ax = sns.barplot(x=['Minimum Age', 'Maximum Age'], y=[threeData['Age'].min(), threeData['Age'].max()])
    ax.bar_label(container=ax.containers[0], labels=[threeData['Age'].min(), threeData['Age'].max()])
    plt.title('Cluster Three')

    plt.subplot(2, 3, 4)
    ax = sns.barplot(x=['Minimum Age', 'Maximum Age'], y=[fourData['Age'].min(), fourData['Age'].max()])
    ax.bar_label(container=ax.containers[0], labels=[fourData['Age'].min(), fourData['Age'].max()])
    plt.title('Cluster Four')

    plt.subplot(2, 3, 5)
    ax = sns.barplot(x=['Minimum Age', 'Maximum Age'], y=[fiveData['Age'].min(), fiveData['Age'].max()])
    ax.bar_label(container=ax.containers[0], labels=[fiveData['Age'].min(), fiveData['Age'].max()])
    plt.title('Cluster Five')
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    plt.show()

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    ax = sns.countplot(x='Gender', data=oneData)
    ax.bar_label(container=ax.containers[0], labels=oneData['Gender'].value_counts(ascending=True))
    plt.title('Cluster One')

    plt.subplot(2, 3, 2)
    ax = sns.countplot(x='Gender', data=twoData)
    ax.bar_label(container=ax.containers[0], labels=twoData['Gender'].value_counts(ascending=True))
    plt.title('Cluster Two')

    plt.subplot(2, 3, 3)
    ax = sns.countplot(x='Gender', data=threeData)
    ax.bar_label(container=ax.containers[0], labels=threeData['Gender'].value_counts(ascending=True))
    plt.title('Cluster Three')

    plt.subplot(2, 3, 4)
    ax = sns.countplot(x='Gender', data=fourData)
    ax.bar_label(container=ax.containers[0], labels=fourData['Gender'].value_counts(ascending=True))
    plt.title('Cluster Four')

    plt.subplot(2, 3, 5)
    ax = sns.countplot(x='Gender', data=fiveData)
    ax.bar_label(container=ax.containers[0], labels=fiveData['Gender'].value_counts(ascending=True))
    plt.title('Cluster Five')
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    plt.show()

# Create the main window
login_window = tk.Tk()
login_window.title("Login/Register")
login_window.state('zoomed')  # Maximize window

# Get the screen width and height
screen_width = login_window.winfo_screenwidth()
screen_height = login_window.winfo_screenheight()

# Calculate the desired tab width and height
tab_width = int(screen_width * 0.8)
tab_height = int(screen_height * 0.8)

# Create a tab control
tab_control = ttk.Notebook(login_window, width=tab_width, height=tab_height)

# Create login tab
login_tab = ttk.Frame(tab_control)
tab_control.add(login_tab, text='Login')

# Create register tab
register_tab = ttk.Frame(tab_control)
tab_control.add(register_tab, text='Register')

# Create browse tab
browse_tab = ttk.Frame(tab_control)
tab_control.add(browse_tab, text='Browse CSV')

# Position the tab control in the center of the screen
x_position = (screen_width - tab_width) // 2
y_position = (screen_height - tab_height) // 2
tab_control.place(x=x_position, y=y_position)

# Login tab components
login_label = tk.Label(login_tab, text="Login", font=("Arial", 24))
username_label = tk.Label(login_tab, text="Username:", font=("Arial", 16))
password_label = tk.Label(login_tab, text="Password:", font=("Arial", 16))
username_entry = tk.Entry(login_tab, font=("Arial", 16))
password_entry = tk.Entry(login_tab, show="*", font=("Arial", 16))
login_button = tk.Button(login_tab, text="Login", command=login, width=30, font=("Arial", 16))
register_button = tk.Button(login_tab, text="Register", command=switch_to_register, width=30, font=("Arial", 16))

login_label.pack()
username_label.pack()
username_entry.pack()
password_label.pack()
password_entry.pack()
login_button.pack()
register_button.pack()

# Register tab components
register_label = tk.Label(register_tab, text="Register", font=("Arial", 24))
register_username_label = tk.Label(register_tab, text="Username:", font=("Arial", 16))
register_password_label = tk.Label(register_tab, text="Password:", font=("Arial", 16))
register_username_entry = tk.Entry(register_tab, font=("Arial", 16))
register_password_entry = tk.Entry(register_tab, show="*", font=("Arial", 16))
register_submit_button = tk.Button(register_tab, text="Submit", command=register, width=30, font=("Arial", 16))

register_label.pack()
register_username_label.pack()
register_username_entry.pack()
register_password_label.pack()
register_password_entry.pack()
register_submit_button.pack()

# Browse tab components
browse_label = tk.Label(browse_tab, text="Browse CSV File", font=("Arial", 24))
browse_button = tk.Button(browse_tab, text="Browse", command=browse_file, width=30, font=("Arial", 16))

browse_label.pack()
browse_button.pack()

# Initially show the login tab
tab_control.select(0)

# Start the GUI event loop
login_window.mainloop()

