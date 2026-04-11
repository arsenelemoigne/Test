import { Resend } from 'resend'

function getResend() {
  return new Resend(process.env.RESEND_API_KEY || 'placeholder')
}

function getFrom() {
  return process.env.RESEND_FROM_EMAIL || 'leads@meshly.com'
}

function getAppUrl() {
  return process.env.NEXT_PUBLIC_APP_URL || 'https://meshly.com'
}

export async function sendWelcomeEmail({
  to,
  businessName,
  profileUrl: profileUrlOverride,
}: {
  to: string
  businessName: string
  profileUrl?: string
}) {
  const APP_URL = getAppUrl()
  const profileUrl = profileUrlOverride ?? `${APP_URL}/profile`

  return getResend().emails.send({
    from: `Meshly <${getFrom()}>`,
    to,
    subject: `Your Meshly profile is live ✓`,
    html: `
<!DOCTYPE html>
<html>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 560px; margin: 0 auto; padding: 40px 20px; color: #111;">
  <div style="margin-bottom: 32px;">
    <span style="font-size: 20px; font-weight: 700; letter-spacing: -0.5px;">Meshly</span>
  </div>

  <p style="font-size: 16px; line-height: 1.6; margin-bottom: 16px;">Hi,</p>

  <p style="font-size: 16px; line-height: 1.6; margin-bottom: 16px;">
    Your AI profile for <strong>${businessName}</strong> is ready.
  </p>

  <p style="font-size: 16px; line-height: 1.6; margin-bottom: 16px;">
    We're now scanning our network for relevant contacts.<br>
    We'll email you directly when we find one.
  </p>

  <p style="font-size: 16px; line-height: 1.6; margin-bottom: 32px;">
    Nothing to do. We'll be in touch.
  </p>

  <p style="font-size: 14px; color: #666; margin-bottom: 8px;">
    P.S. You can strengthen your profile to get better matches:
  </p>

  <a href="${profileUrl}"
     style="display: inline-block; background: #111; color: #fff; padding: 12px 24px; border-radius: 8px; text-decoration: none; font-size: 14px; font-weight: 500;">
    Complete my profile →
  </a>

  <div style="margin-top: 48px; padding-top: 24px; border-top: 1px solid #eee; font-size: 13px; color: #999;">
    Meshly · <a href="${APP_URL}/unsubscribe?email=${encodeURIComponent(to)}" style="color: #999;">Unsubscribe</a>
  </div>
</body>
</html>`,
    text: `Hi,

Your AI profile for ${businessName} is ready.

We're now scanning our network for relevant contacts. We'll email you directly when we find one.

Nothing to do. We'll be in touch.

— Meshly

P.S. You can strengthen your profile to get better matches: ${profileUrl}`,
  })
}

export async function sendLeadEmail({
  to,
  recipientBusinessName,
  senderBusiness,
  matchReason,
  matchScore,
  leadId,
}: {
  to: string
  recipientBusinessName: string
  senderBusiness: {
    name: string
    industry: string
    location: string
    offering_summary: string
  }
  matchReason: string
  matchScore: number
  leadId: string
}) {
  const APP_URL = getAppUrl()
  const interestedUrl = `${APP_URL}/api/leads/${leadId}/interested`
  const declinedUrl = `${APP_URL}/api/leads/${leadId}/declined`
  const scorePercent = Math.round(matchScore * 100)
  const scoreDots = '●'.repeat(Math.round(matchScore * 5)) + '○'.repeat(5 - Math.round(matchScore * 5))

  return getResend().emails.send({
    from: `Meshly <${getFrom()}>`,
    to,
    subject: `A ${senderBusiness.industry} company in ${senderBusiness.location} might interest you`,
    html: `
<!DOCTYPE html>
<html>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 560px; margin: 0 auto; padding: 40px 20px; color: #111;">
  <div style="margin-bottom: 32px;">
    <span style="font-size: 20px; font-weight: 700; letter-spacing: -0.5px;">Meshly</span>
  </div>

  <p style="font-size: 16px; line-height: 1.6; margin-bottom: 24px;">
    Hi ${recipientBusinessName},
  </p>

  <p style="font-size: 16px; line-height: 1.6; margin-bottom: 24px;">
    We found a potential contact for you.
  </p>

  <div style="border: 1px solid #e5e5e5; border-radius: 12px; padding: 24px; margin-bottom: 24px;">
    <p style="font-size: 18px; font-weight: 600; margin: 0 0 8px;">${senderBusiness.name}</p>
    <p style="font-size: 14px; color: #666; margin: 0 0 16px;">${senderBusiness.industry} · ${senderBusiness.location}</p>

    <p style="font-size: 14px; color: #444; font-style: italic; margin: 0 0 16px;">
      "${senderBusiness.offering_summary}"
    </p>

    <p style="font-size: 13px; color: #666; margin: 0 0 8px;">
      <strong>Why we think this is relevant:</strong><br>
      ${matchReason}
    </p>

    <p style="font-size: 13px; color: #666; margin: 0;">
      Match strength: ${scoreDots} ${scorePercent}%
    </p>
  </div>

  <table style="width: 100%; border-collapse: separate; border-spacing: 8px;">
    <tr>
      <td style="width: 50%;">
        <a href="${interestedUrl}"
           style="display: block; text-align: center; background: #111; color: #fff; padding: 14px; border-radius: 8px; text-decoration: none; font-size: 14px; font-weight: 500;">
          ★ I'm interested
        </a>
      </td>
      <td style="width: 50%;">
        <a href="${declinedUrl}"
           style="display: block; text-align: center; background: #f5f5f5; color: #666; padding: 14px; border-radius: 8px; text-decoration: none; font-size: 14px;">
          ✗ Not relevant
        </a>
      </td>
    </tr>
  </table>

  <p style="font-size: 12px; color: #aaa; text-align: center; margin-top: 16px;">
    Clicking "Interested" will show you their contact details
  </p>

  <div style="margin-top: 48px; padding-top: 24px; border-top: 1px solid #eee; font-size: 13px; color: #999;">
    This lead was found automatically by Meshly.<br>
    <a href="${APP_URL}/settings" style="color: #999;">Manage preferences</a> ·
    <a href="${APP_URL}/api/leads/${leadId}/declined" style="color: #999;">Unsubscribe</a>
  </div>
</body>
</html>`,
    text: `Hi ${recipientBusinessName},

We found a potential contact for you.

${senderBusiness.name}
${senderBusiness.industry} · ${senderBusiness.location}

"${senderBusiness.offering_summary}"

Why we think this is relevant:
${matchReason}

Match strength: ${scorePercent}%

I'm interested: ${interestedUrl}
Not relevant: ${declinedUrl}

—
This lead was found automatically by Meshly.`,
  })
}
