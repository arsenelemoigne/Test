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
  selected?: string | null;
  onSelect?: (code: string | null) => void;
  // Wording for the tooltip, e.g. "titre(s)" or "signalement(s)".
  unit: string;
};

export function WorldMap({ counts, selected, onSelect, unit }: Props) {
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

  const max = Math.max(1, ...Array.from(counts.values()));
  // Countries above half the maximum are black, the others dark grey.
  const half = Math.floor(max / 2);

  return (
    <div className="relative w-full">
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="h-auto w-full" role="img" aria-label="Carte du monde">
        {paths.map((p, i) => {
          const active = p.n > 0;
          const isSelected = selected && p.code === selected;
          // Two shades only: covered countries in grey to black by count, others in light grey.
          const fill = active ? (p.n > half ? "#000" : "#555") : "#e8e8e8";
          return (
            <path
              key={`${p.code}-${i}`}
              d={p.d}
              fill={fill}
              stroke={isSelected ? "#000" : "#fff"}
              strokeWidth={isSelected ? 1.6 : 0.4}
              className={onSelect && p.code ? "cursor-pointer hover:opacity-80" : undefined}
              onClick={() => onSelect?.(p.code && p.code !== selected ? p.code : null)}
            >
              <title>{p.code ? `${countryName(p.code)} — ${p.n} ${unit}` : ""}</title>
            </path>
          );
        })}
      </svg>
      {!features && <div className="absolute inset-0 grid place-items-center font-mono text-xs text-neutral-500">Chargement de la carte…</div>}
      <div className="mt-2 flex items-center gap-4 font-mono text-[11px] text-neutral-600">
        <span className="flex items-center gap-1.5"><span className="inline-block h-2.5 w-2.5 bg-black" /> {half + 1} {unit} ou plus</span>
        {half >= 1 && <span className="flex items-center gap-1.5"><span className="inline-block h-2.5 w-2.5 bg-[#555]" /> {half === 1 ? "1" : `1 à ${half}`}</span>}
        <span className="flex items-center gap-1.5"><span className="inline-block h-2.5 w-2.5 bg-[#e8e8e8]" /> aucun</span>
      </div>
    </div>
  );
}
