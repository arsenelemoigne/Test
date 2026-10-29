"use client";

import { useEffect, useMemo, useState } from "react";
import { geoNaturalEarth1, geoPath } from "d3-geo";
import { feature } from "topojson-client";
import type { Feature, FeatureCollection, Geometry } from "geojson";
import type { Topology } from "topojson-specification";
import { countryName, numericToAlpha2 } from "@/lib/territories";

type CountryFeature = Feature<Geometry, { name: string }> & { id?: string | number };

let atlas: Promise<CountryFeature[]> | null = null;

function loadAtlas(): Promise<CountryFeature[]> {
  if (!atlas) {
    atlas = fetch("/world-50m.json")
      .then((r) => r.json())
      .then((topo: Topology) => {
        const fc = feature(topo, topo.objects.countries) as unknown as FeatureCollection<Geometry, { name: string }>;
        // Antarctica takes space without carrying any rights.
        return (fc.features as CountryFeature[]).filter((f) => String(f.id) !== "010");
      });
  }
  return atlas;
}

const WIDTH = 960;
const HEIGHT = 470;

type Props = {
  counts: Map<string, number>;
  // "coverage": countries where the marks apply (hatched); "hits": countries of search results (inked).
  mode: "coverage" | "hits";
  selected?: string | null;
  onSelect?: (code: string | null) => void;
  unit: string;
};

export function WorldMap({ counts, mode, selected, onSelect, unit }: Props) {
  const [features, setFeatures] = useState<CountryFeature[] | null>(null);

  useEffect(() => {
    loadAtlas().then(setFeatures).catch(() => setFeatures([]));
  }, []);

  const paths = useMemo(() => {
    if (!features) return [];
    const projection = geoNaturalEarth1().fitSize([WIDTH, HEIGHT], { type: "FeatureCollection", features } as FeatureCollection);
    const path = geoPath(projection);
    return features.map((f) => {
      const code = numericToAlpha2(f.id) ?? "";
      return { code, d: path(f) ?? "", n: counts.get(code) ?? 0 };
    });
  }, [features, counts]);

  const fillFor = (n: number) => {
    if (n === 0) return "#ebe6dc";
    return mode === "coverage" ? "url(#ipw-hatch)" : "#1c1c1a";
  };

  return (
    <div className="relative w-full">
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="h-auto w-full" role="img" aria-label="Carte du monde">
        <defs>
          <pattern id="ipw-hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(40)">
            <rect width="5" height="5" fill="#d8d1c3" />
            <line x1="0" y1="0" x2="0" y2="5" stroke="#1c1c1a" strokeWidth="1.6" />
          </pattern>
        </defs>
        {paths.map((p, i) => {
          const isSelected = selected && p.code === selected;
          return (
            <path
              key={`${p.code}-${i}`}
              d={p.d}
              fill={fillFor(p.n)}
              stroke={isSelected ? "#1c1c1a" : p.n ? "#1c1c1a" : "#faf8f4"}
              strokeWidth={isSelected ? 1.8 : p.n ? 0.5 : 0.6}
              strokeLinejoin="round"
              className={onSelect && p.code ? "cursor-pointer transition-opacity hover:opacity-75" : undefined}
              onClick={() => onSelect?.(p.code && p.code !== selected ? p.code : null)}
            >
              <title>{p.code ? `${countryName(p.code)}${p.n ? ` — ${p.n} ${unit}` : ""}` : ""}</title>
            </path>
          );
        })}
      </svg>
      {!features && <div className="hand absolute inset-0 grid place-items-center text-sm text-[color:var(--pencil)]">la carte se dessine…</div>}
      <div className="label mt-1 flex items-center gap-4">
        <span className="flex items-center gap-1.5">
          <svg width="14" height="10" aria-hidden>
            <rect width="14" height="10" fill={mode === "coverage" ? "url(#ipw-hatch)" : "#1c1c1a"} stroke="#1c1c1a" strokeWidth="1" />
          </svg>
          {mode === "coverage" ? "protégé" : unit}
        </span>
        <span className="flex items-center gap-1.5">
          <svg width="14" height="10" aria-hidden>
            <rect width="14" height="10" fill="#ebe6dc" />
          </svg>
          {mode === "coverage" ? "non couvert" : "aucun"}
        </span>
      </div>
    </div>
  );
}
