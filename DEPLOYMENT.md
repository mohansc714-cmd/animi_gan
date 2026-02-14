# 🚀 Deployment Guide - Get Your Public URL

## Option 1: Hugging Face Spaces (Recommended - FREE & Permanent)

### Step 1: Create Hugging Face Account
1. Go to https://huggingface.co/join
2. Sign up for a free account

### Step 2: Create a New Space
1. Go to https://huggingface.co/new-space
2. Fill in:
   - **Space name**: `anime-gan-generator` (or your choice)
   - **License**: MIT
   - **SDK**: Docker
   - **Space hardware**: CPU basic (free)
3. Click "Create Space"

### Step 3: Upload Your Files
You need to upload these files to your Space:
- ✅ `app.py`
- ✅ `model.py`
- ✅ `requirements.txt`
- ✅ `Dockerfile`
- ✅ `README.md`
- ✅ `templates/index.html`
- ✅ `generator_final.keras` (your trained model - IMPORTANT!)

**Two ways to upload:**

#### Method A: Using Git (Recommended)
```bash
# In your project directory
git init
git add app.py model.py requirements.txt Dockerfile README.md templates/ generator_final.keras
git commit -m "Initial commit"

# Add your Hugging Face Space as remote (replace USERNAME and SPACENAME)
git remote add origin https://huggingface.co/spaces/USERNAME/SPACENAME
git push origin main
```

#### Method B: Using Web Interface
1. Click "Files" tab in your Space
2. Click "Add file" → "Upload files"
3. Drag and drop all the files listed above
4. Click "Commit changes to main"

### Step 4: Wait for Build
- The Space will automatically build (takes 5-10 minutes)
- You'll see build logs in the "Logs" tab
- Once complete, your app will be live!

### Step 5: Get Your URL
Your public URL will be:
```
https://huggingface.co/spaces/YOUR-USERNAME/YOUR-SPACE-NAME
```

---

## Option 2: Local Access with ngrok (Quick Temporary URL)

### Step 1: Install ngrok
1. Download from: https://ngrok.com/download
2. Extract and place ngrok.exe in your project folder
3. Sign up at https://ngrok.com to get auth token
4. Run: `ngrok config add-authtoken YOUR_TOKEN`

### Step 2: Start Your App
```bash
python app.py
```

### Step 3: Create Tunnel
In a new terminal:
```bash
ngrok http 7860
```

### Step 4: Get Your URL
ngrok will show you a URL like:
```
https://abc123.ngrok.io
```
Share this URL! (Note: It changes each time you restart ngrok)

---

## Option 3: Render.com (Free Permanent Hosting)

### Step 1: Create Account
1. Go to https://render.com
2. Sign up with GitHub

### Step 2: Create Web Service
1. Click "New +" → "Web Service"
2. Connect your GitHub repo
3. Configure:
   - **Name**: anime-gan
   - **Environment**: Docker
   - **Plan**: Free
4. Click "Create Web Service"

### Step 3: Get Your URL
Your URL will be:
```
https://anime-gan.onrender.com
```

---

## 📝 Important Notes

1. **Model File Size**: `generator_final.keras` is ~11MB. Make sure to upload it!

2. **Training**: The current model was trained for 25 epochs at 64x64. 
   - For better quality, wait for the 100-epoch 128x128 training to complete
   - Then replace `generator_final.keras` with the new model

3. **Free Tier Limitations**:
   - Hugging Face: CPU only, but sufficient for inference
   - Render: May sleep after inactivity, takes 30s to wake up
   - ngrok: Temporary URLs, resets on restart

4. **Recommended**: Use Hugging Face Spaces for the best free permanent solution!

---

## 🎯 Quick Start (Hugging Face)

If you have git configured:

```bash
cd c:\Users\student\Desktop\animi_gan\animi_gan

# Initialize git
git init
git add app.py model.py requirements.txt Dockerfile README.md templates/ generator_final.keras
git commit -m "Deploy anime GAN"

# Push to Hugging Face (replace with your details)
git remote add origin https://huggingface.co/spaces/YOUR-USERNAME/anime-gan
git push origin main
```

Your app will be live at: `https://huggingface.co/spaces/YOUR-USERNAME/anime-gan`

---

## ❓ Need Help?

If you encounter issues:
1. Check the build logs in Hugging Face Spaces
2. Ensure `generator_final.keras` is uploaded
3. Verify all files are present
4. Check that requirements.txt has all dependencies
