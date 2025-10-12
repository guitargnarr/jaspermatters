# ✅ GoDaddy DNS Configuration Checklist

## Records to DELETE (if they exist):
□ A record @ → 18.208.88.157 (Netlify)
□ A record @ → 98.84.224.111 (Netlify)
□ Any other A records with @ pointing elsewhere
□ Any CNAME pointing to netlify.app or netlify.com

## Records to ADD:
□ A record @ → 185.199.108.153 (GitHub)
□ A record @ → 185.199.109.153 (GitHub)
□ A record @ → 185.199.110.153 (GitHub)
□ A record @ → 185.199.111.153 (GitHub)
□ CNAME www → guitargnar.github.io

## After DNS Update:
□ Wait 2-3 minutes
□ Go to GitHub Pages settings
□ Click "Check again" next to DNS error
□ Wait for "DNS check successful" ✅
□ Enable "Enforce HTTPS"
□ Wait 5-10 minutes for deployment

## Verify It's Working:
```bash
# Run this command to check DNS
nslookup jaspermatters.com

# Should show:
# 185.199.108.153
# 185.199.109.153
# 185.199.110.153
# 185.199.111.153

# Test the site (after 10 minutes)
open https://jaspermatters.com
```

## Timeline:
- DNS update in GoDaddy: 2 minutes
- DNS propagation: 5-15 minutes
- GitHub verification: 2-3 minutes
- Site deployment: 5-10 minutes
- **Total: ~15-30 minutes**

## Troubleshooting:

**If DNS check still fails after 15 minutes:**
- Make sure you deleted ALL old A records
- Verify all 4 GitHub IPs are added
- Clear browser cache and try again

**If site shows 404:**
- DNS is working but GitHub Pages needs more time
- Check: https://github.com/guitargnar/jaspermatters-job-intelligence/actions
- Look for green checkmark on "pages build and deployment"

**If still showing OurJourney:**
- DNS hasn't propagated yet
- Try incognito/private browser mode
- Clear browser cache (Cmd+Shift+R)

---

**Your ML Platform is ready to deploy - just waiting for DNS!** 🚀