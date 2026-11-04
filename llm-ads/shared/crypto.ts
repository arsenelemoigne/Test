import {
  createHmac,
  createPrivateKey,
  createPublicKey,
  generateKeyPairSync,
  sign,
  timingSafeEqual,
  verify,
} from 'node:crypto';
import type { ClickParams } from './types.ts';

/** JSON with sorted keys, so both sides sign and verify the exact same bytes. */
export function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  const entries = Object.entries(value as Record<string, unknown>)
    .filter(([, v]) => v !== undefined)
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
  return `{${entries.map(([k, v]) => `${JSON.stringify(k)}:${canonicalJson(v)}`).join(',')}}`;
}

export function generateCatalogKeys(): { privateKeyPem: string; publicKeyPem: string } {
  const { privateKey, publicKey } = generateKeyPairSync('ed25519');
  return {
    privateKeyPem: privateKey.export({ type: 'pkcs8', format: 'pem' }).toString(),
    publicKeyPem: publicKey.export({ type: 'spki', format: 'pem' }).toString(),
  };
}

export function publicKeyFromPrivate(privateKeyPem: string): string {
  return createPublicKey(createPrivateKey(privateKeyPem)).export({ type: 'spki', format: 'pem' }).toString();
}

export function signPayload(payload: unknown, privateKeyPem: string): string {
  return sign(null, Buffer.from(canonicalJson(payload)), createPrivateKey(privateKeyPem)).toString('base64');
}

export function verifyPayload(payload: unknown, signature: string, publicKeyPem: string): boolean {
  try {
    return verify(null, Buffer.from(canonicalJson(payload)), createPublicKey(publicKeyPem), Buffer.from(signature, 'base64'));
  } catch {
    return false;
  }
}

export function hmacHex(secret: string, message: string): string {
  return createHmac('sha256', secret).update(message).digest('hex');
}

export function safeEqualHex(a: string, b: string): boolean {
  const ba = Buffer.from(a, 'hex');
  const bb = Buffer.from(b, 'hex');
  return ba.length > 0 && ba.length === bb.length && timingSafeEqual(ba, bb);
}

/** The exact string a publisher signs for a click link. Ids are validated to contain no "|". */
export function clickMessage(p: ClickParams): string {
  return ['click', p.offerId, p.publisherId, p.ts, p.nonce].join('|');
}

export const ID_PATTERN = /^[a-z0-9][a-z0-9-]{0,63}$/;
