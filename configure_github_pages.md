# 🔧 Configure GitHub Pages - Step by Step

## Quick Setup (5 minutes)

I'll guide you through configuring GitHub Pages. Follow these steps exactly:

### Step 1: Open GitHub Settings
```bash
# Copy and paste this URL into your browser:
https://github.com/guitargnarr/jaspermatters/settings/pages
```

### Step 2: Configure Source
When the page loads, you'll see "GitHub Pages" settings:

1. **Build and deployment**
   - Source: Select `Deploy from a branch` ✓
   
2. **Branch**
   - Branch: Select `main` (should be default)
   - Folder: Select `/ (root)`
   - Click **Save**

### Step 3: Add Custom Domain
After saving, you'll see a new section appear:

1. **Custom domain**
   - Type: `jaspermatters.com`
   - Click **Save**
   
2. **Wait 1-2 minutes**, then:
   - ✅ Check "Enforce HTTPS" (this option appears after domain verification)

### Step 4: Check Deployment Status
```bash
# Visit this URL to see build status:
https://github.com/guitargnarr/jaspermatters/actions
```

You should see a workflow running called "pages build and deployment"

### Step 5: Configure DNS (at your domain registrar)

Go to your domain registrar (GoDaddy, Namecheap, etc.) and add these records:

```
Type    Host/Name    Points to/Value           TTL
A       @           185.199.108.153           600
A       @           185.199.109.153           600  
A       @           185.199.110.153           600
A       @           185.199.111.153           600
CNAME   www         guitargnarr.github.io.     600
```

**Important**: Delete any existing A records that point elsewhere!

## 🎯 Verification Steps

### Check GitHub Pages is Active:
1. Go back to: https://github.com/guitargnarr/jaspermatters/settings/pages
2. You should see: "✅ Your site is live at https://jaspermatters.com"

### Test Your Site (after DNS propagates):
1. https://jaspermatters.com - Landing page with ML demos
2. https://jaspermatters.com/resume.html - Your visual resume
3. Try the terminal commands (type "help")

### DNS Propagation Check:
```bash
# Run this command to check if DNS is working:
nslookup jaspermatters.com

# You should see GitHub's IPs (185.199.xxx.xxx)
```

## 🚨 Troubleshooting

### "404 - There isn't a GitHub Pages site here"
- Make sure you clicked **Save** after selecting the branch
- Wait 5-10 minutes for initial deployment
- Check that index.html exists in the main branch ✓

### Custom domain not working:
- DNS can take 24-48 hours to fully propagate
- Try: https://guitargnarr.github.io/jaspermatters-job-intelligence first
- If that works, DNS just needs more time

### HTTPS not available:
- This appears only after domain verification (can take 10-15 minutes)
- Keep refreshing the settings page
- It will appear automatically

## ✅ Success Indicators

You'll know it's working when:
1. Settings page shows: "✅ Your site is live at https://jaspermatters.com"
2. Actions tab shows: "✅ pages build and deployment" completed
3. You can access: https://guitargnarr.github.io/jaspermatters-job-intelligence
4. Eventually: https://jaspermatters.com loads your site

## 📱 Quick Mobile Test

Once configured, test on your phone:
1. Open jaspermatters.com
2. Try the interactive demos
3. Check resume page formatting
4. Share with a friend for feedback!

---

**Estimated Time**: 
- GitHub setup: 5 minutes ⚡
- DNS propagation: 10 minutes to 24 hours 🌍

**Status Check URL**: https://github.com/guitargnarr/jaspermatters/settings/pages