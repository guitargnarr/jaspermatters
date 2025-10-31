# Google Analytics Setup Instructions

## Step 1: Create GA4 Property

1. Go to: https://analytics.google.com
2. Click "Admin" (bottom left)
3. Click "Create Property"
4. Name: "JasperMatters ML Platform"
5. Select timezone and currency
6. Click "Next"
7. Choose business details (optional)
8. Click "Create"

## Step 2: Get Measurement ID

1. In Property settings, click "Data Streams"
2. Click "Add stream" → "Web"
3. Website URL: `https://jaspermatters.com`
4. Stream name: "JasperMatters Production"
5. Click "Create stream"
6. **Copy the Measurement ID** (looks like `G-XXXXXXXXXX`)

## Step 3: Add to Website

Once you have your Measurement ID, edit `index.html`:

Replace this section:
```html
<!-- Google Analytics (placeholder - add your GA4 ID) -->
<!-- <script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script> -->
<script>
  // window.dataLayer = window.dataLayer || [];
  // function gtag(){dataLayer.push(arguments);}
  // gtag('js', new Date());
  // gtag('config', 'G-XXXXXXXXXX');
</script>
```

With:
```html
<!-- Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-YOUR-ID-HERE"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-YOUR-ID-HERE');
</script>
```

## Step 4: Redeploy

```bash
cd /tmp/jaspermatters
git add index.html
git commit -m "feat(analytics): Add Google Analytics tracking"
git push origin main
netlify deploy --prod --dir=dist
```

## Step 5: Verify Tracking

1. Visit https://jaspermatters.com
2. Go back to GA4 → Reports → Realtime
3. You should see 1 active user (you!)

## What You'll Track

- Page views
- Demo usage (which demos people try)
- Time on site
- Geographic location of visitors
- Device types (mobile vs desktop)
- Referral sources (LinkedIn, direct, etc.)

## Privacy Note

GA4 complies with GDPR. Consider adding a privacy policy page if you get significant traffic, but for a portfolio demo site, the disclaimer in the footer is sufficient.
