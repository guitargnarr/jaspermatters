# 🔄 Switch DNS from Netlify to GitHub Pages

## Current Status
- Domain: jaspermatters.com
- Currently pointing to: Netlify (18.208.88.157, 98.84.224.111)
- Need to point to: GitHub Pages

## Step 1: Remove from Netlify

### Option A: Via Netlify Dashboard
1. Go to: https://app.netlify.com
2. Check "Sites" tab for any site using jaspermatters.com
3. If found: Site Settings → Domain Management → Remove custom domain
4. Check "Domains" tab for registered domains
5. If found: Click domain → Delete/Remove

### Option B: Via Domain Registrar
Go to your registrar (GoDaddy, Namecheap, Google Domains, etc.)

**DELETE these records:**
```
A     @     18.208.88.157     ← Delete this
A     @     98.84.224.111     ← Delete this
CNAME www   [anything].netlify.app  ← Delete if exists
```

**ADD these records:**
```
Type    Host/Name    Value                   TTL
A       @           185.199.108.153         600
A       @           185.199.109.153         600
A       @           185.199.110.153         600
A       @           185.199.111.153         600
CNAME   www         guitargnarr.github.io.   600
```

## Step 2: Verify in GitHub Pages

After updating DNS:
1. Go back to GitHub Pages settings
2. Click "Check again" button next to DNS error
3. Wait for green checkmark ✅
4. Once verified, check "Enforce HTTPS"

## Step 3: Test Deployment

```bash
# Check DNS (should show GitHub IPs)
nslookup jaspermatters.com

# Should return:
# 185.199.108.153
# 185.199.109.153
# 185.199.110.153
# 185.199.111.153

# Test site
curl -I https://jaspermatters.com

# Should return HTTP 200 with your content
```

## Timeline
- Netlify removal: Instant
- DNS propagation: 5-30 minutes
- GitHub Pages build: 5 minutes after DNS verified
- HTTPS certificate: 10-15 minutes after domain verified

## Common Registrars - Where to Update

### GoDaddy
My Products → Domains → jaspermatters.com → DNS → Manage Zones

### Namecheap
Domain List → jaspermatters.com → Advanced DNS

### Google Domains
My domains → jaspermatters.com → DNS

### Cloudflare
DNS → Records → Edit/Delete A records, Add new ones

## Success Checklist
□ Old Netlify A records removed
□ GitHub Pages A records added
□ CNAME for www added
□ GitHub Pages shows "DNS check successful"
□ "Enforce HTTPS" enabled
□ jaspermatters.com shows ML platform (not OurJourney)

---

**Once DNS is updated, your Job Intelligence Platform will automatically deploy!**