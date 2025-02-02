# 🚀 Installing and Running Jupyter Notebook in GitHub Codespaces

Welcome to the guide on setting up Jupyter Notebook within **GitHub Codespaces**! Follow these simple steps to install and access Jupyter Notebook.

---

## 1. **Install JupyterLab and Notebook** 🖥️

To begin, you'll need to install both **JupyterLab** and **Jupyter Notebook** in your Codespace. Simply run the following command in the terminal:

```bash
pip install jupyterlab notebook
```

This will install:
- **JupyterLab**: A modern interface for working with Jupyter Notebooks.
- **Jupyter Notebook**: The classic environment for running and sharing code.

---

## 2. **Launch Jupyter Notebook Server** 🚀

After installation, you can start the Jupyter Notebook server with the following command:

```bash
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
```

### Explanation:
- `--ip=0.0.0.0`: Allows access from any IP address.
- `--port=8888`: Runs Jupyter on port 8888.
- `--no-browser`: Prevents Jupyter from opening in a browser automatically.

---

## 3. **Get Your Access Token** 🔑

Once the server starts, you’ll see an output with a URL that contains a **token**. It will look something like this:

```
    To access the notebook, open this file in a browser:
        file:///home/username/.local/share/jupyter/runtime/nbserver-1234-open.html
    Or copy and paste one of these URLs:
        http://127.0.0.1:8888/?token=00462e7b4bd20e6cd2057680e7cdb92d4a8c127c0f5b93ba
```

#### Example Token URL:
```
http://127.0.0.1:8888/tree?token=00462e7b4bd20e6cd2057680e7cdb92d4a8c127c0f5b93ba
```

---

## 4. **Access Jupyter Notebook** 🌐

### Steps:
1. Copy the URL containing your **token**.
2. Paste it into your browser's address bar.
3. You're now in the Jupyter Notebook interface! 🎉

Now you can start creating and running notebooks for your data science, machine learning, and Python projects!

---

## ⚠️ Troubleshooting

If you face any issues, here are some quick fixes:

### **1. URL Not Opening in Browser?**
- Make sure the `token` is correctly copied into the URL.

### **2. Token Expired?**
- You can regenerate your token by running:

```bash
jupyter notebook list
```

This will display the active token and the correct URL.

---

## ✨ Conclusion

By following these steps, you’ll have Jupyter Notebook running smoothly in **GitHub Codespaces**. Whether you’re working on machine learning, data analysis, or just playing around with Python, you now have a powerful tool at your fingertips!

Happy coding! 🎉

```

### Key Visual Enhancements:
- **Headings** are styled to make each section easy to locate.
- **Emojis** were added to make the guide more engaging.
- **Code blocks** are clearly highlighted for easy copying.
- **Bold text** was used to emphasize key points like commands and important steps.