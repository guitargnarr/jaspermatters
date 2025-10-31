# 🚀 Migrating from Render to GitHub Pages

## Current Status
✅ Found: ourjourney-api on Render using jaspermatters.com
✅ Issue: Render controls DNS via NS1 nameservers
✅ Goal: Point jaspermatters.com to GitHub Pages for ML platform

## Step-by-Step Migration

### 1️⃣ Remove Domain from Render (In Progress)
- Click on **ourjourney-api** service
- Go to **Settings** → **Custom Domains**
- Remove `jaspermatters.com`
- Optionally delete the service if not needed

### 2️⃣ Switch Nameservers in GoDaddy
- Go to GoDaddy Nameservers tab
- Click **Change**
- Select **GoDaddy Nameservers**:
  - ns1.godaddy.com
  - ns2.godaddy.com
- Save changes

### 3️⃣ Configure DNS for GitHub Pages
Wait 5 minutes after nameserver change, then in GoDaddy DNS Records:

**Add these A records:**
```
Type: A    Name: @    Value: 185.199.108.153    TTL: 600
Type: A    Name: @    Value: 185.199.109.153    TTL: 600
Type: A    Name: @    Value: 185.199.110.153    TTL: 600
Type: A    Name: @    Value: 185.199.111.153    TTL: 600
```

**Add this CNAME:**
```
Type: CNAME    Name: www    Value: guitargnarr.github.io    TTL: 600
```

### 4️⃣ Verify in GitHub Pages
- Go to: https://github.com/guitargnarr/jaspermatters/settings/pages
- Click "Check again" next to DNS error
- Once green ✅, enable "Enforce HTTPS"

## Timeline
- Render domain removal: Instant
- Nameserver change: 5-30 minutes
- DNS record updates: 5-15 minutes
- GitHub Pages deployment: 5-10 minutes
- **Total: ~30-45 minutes**

## Success Checklist
□ Domain removed from Render
□ Nameservers switched to GoDaddy
□ GitHub A records added
□ CNAME for www added
□ GitHub Pages shows "DNS check successful"
□ jaspermatters.com shows ML platform

## Test Commands
```bash
# Check nameservers (should show godaddy.com)
nslookup -type=NS jaspermatters.com

# Check A records (should show GitHub IPs)
nslookup jaspermatters.com

# Test site
curl -I https://jaspermatters.com
```

---

**Your ML Platform is ready to replace OurJourney!**