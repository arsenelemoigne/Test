import type { Marketplace } from './store.ts';

/** Fictional advertisers and offers for local development. */
export function seedMarketplace(m: Marketplace, publisher: { id: string; secret: string }): void {
  const seed = [
    {
      advertiser: { id: 'stride', name: 'Stride Running Co.', budgetCents: 5_000 },
      offer: {
        id: 'stride-stability',
        title: 'Stability running shoes for flat feet',
        claims: ['Arch-support models for overpronation', 'Free returns within 60 days'],
        keywords: ['running', 'shoes', 'sneakers', 'flat feet', 'overpronation', 'arch support', 'jogging', 'marathon'],
        categories: ['sports.running'],
        promoCode: 'STRIDE15',
        bidCents: 45,
      },
    },
    {
      advertiser: { id: 'trailhead', name: 'Trailhead Outdoor', budgetCents: 3_000 },
      offer: {
        id: 'trailhead-trail',
        title: 'Trail running shoes with grip for mud and rock',
        claims: ['Vibram-style outsoles', 'Waterproof options'],
        keywords: ['trail', 'running', 'shoes', 'hiking', 'mountain', 'mud'],
        categories: ['sports.running', 'outdoor'],
        bidCents: 60,
      },
    },
    {
      advertiser: { id: 'kestrel', name: 'Kestrel VPN', budgetCents: 10_000 },
      offer: {
        id: 'kestrel-vpn',
        title: 'No-logs VPN for public Wi-Fi and travel',
        claims: ['Independent no-logs audit', '30-day money-back guarantee'],
        keywords: ['vpn', 'privacy', 'wifi', 'public wifi', 'streaming', 'security', 'proxy'],
        categories: ['software.security'],
        promoCode: 'KESTREL3M',
        bidCents: 120,
      },
    },
    {
      advertiser: { id: 'lumen', name: 'Lumen Laptops', budgetCents: 4_000 },
      offer: {
        id: 'lumen-student',
        title: 'Lightweight laptops with student pricing',
        claims: ['Student discount with a valid student email', '2-year warranty'],
        keywords: ['laptop', 'notebook', 'computer', 'student', 'programming', 'ultrabook'],
        categories: ['electronics.computers'],
        bidCents: 80,
      },
    },
    {
      advertiser: { id: 'parlo', name: 'Parlo Languages', budgetCents: 2_000 },
      offer: {
        id: 'parlo-lessons',
        title: 'Live language lessons with native tutors',
        claims: ['First lesson free', 'Spanish, French, German and Italian'],
        keywords: ['learn', 'language', 'spanish', 'french', 'german', 'italian', 'tutor', 'lessons'],
        categories: ['education.languages'],
        bidCents: 50,
      },
    },
    {
      advertiser: { id: 'wayfare', name: 'Wayfare Rail', budgetCents: 100 },
      offer: {
        id: 'wayfare-rail',
        title: 'Cheap train tickets across Europe',
        claims: ['Compare rail operators in one search'],
        keywords: ['train', 'rail', 'tickets', 'europe', 'interrail', 'travel'],
        categories: ['travel.rail'],
        bidCents: 35,
      },
    },
  ];

  for (const { advertiser, offer } of seed) {
    m.addAdvertiser(advertiser);
    m.addOffer({ ...offer, advertiserId: advertiser.id, brand: advertiser.name, url: `https://example.com/${offer.id}` });
  }
  m.addPublisher({ id: publisher.id, name: 'Demo chat app', secret: publisher.secret });
}
