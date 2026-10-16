import type React from 'react';
export function LogoMark() {
  return (
    <div className="logo-mark" aria-hidden="true">
      <span />
      <span />
      <span />
    </div>
  );
}

export function NetworkVisual() {
  const nodes = [
    { x: 14, y: 24, label: "PATIENT", icon: "01", color: "#77f2d0" },
    { x: 29, y: 55, label: "FEATURES", icon: "02", color: "#91d7ff" },
    { x: 48, y: 30, label: "STAGE", icon: "03", color: "#ad91ff" },
    { x: 64, y: 62, label: "MODEL", icon: "04", color: "#ad91ff" },
    { x: 82, y: 34, label: "SIGNAL", icon: "05", color: "#f3bd69" },
    { x: 91, y: 76, label: "SHAP", icon: "06", color: "#77f2d0" },
  ];
  return (
    <div className="network-wrap" aria-label="Abstract pipeline visualization: patient data flows through feature processing, cost stage, model, and explanation.">
      <div className="network-orbit orbit-one" />
      <div className="network-orbit orbit-two" />
      <svg className="network-svg" viewBox="0 0 100 100" role="img">
        <defs>
          <linearGradient id="traceMint" x1="0" x2="1">
            <stop offset="0" stopColor="#77f2d0" stopOpacity="0" />
            <stop offset=".5" stopColor="#77f2d0" />
            <stop offset="1" stopColor="#ad91ff" stopOpacity=".1" />
          </linearGradient>
          <filter id="softGlow"><feGaussianBlur stdDeviation="1.1" result="blur" /><feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
        </defs>
        <path d="M 14 24 C 22 24, 22 52, 29 55 S 39 32, 48 30 S 55 59, 64 62 S 75 35, 82 34 S 84 70, 91 76" className="network-trace trace-flow" fill="none" stroke="url(#traceMint)" strokeWidth=".38" strokeDasharray="2.2 2" />
        <path d="M 14 24 C 22 15, 34 14, 48 30 S 70 50, 82 34" className="network-ghost" fill="none" stroke="#78a9a1" strokeOpacity=".18" strokeWidth=".25" />
        <path d="M 29 55 C 40 68, 50 72, 64 62 S 80 71, 91 76" className="network-ghost" fill="none" stroke="#ad91ff" strokeOpacity=".21" strokeWidth=".25" />
        {nodes.map((node, index) => (
          <g key={node.label} className="network-node" style={{ animationDelay: `${index * 180}ms` }}>
            <circle cx={node.x} cy={node.y} r={index === 4 ? 4.6 : 3.6} fill="#091013" stroke={node.color} strokeWidth=".5" filter="url(#softGlow)" />
            <circle cx={node.x} cy={node.y} r="1" fill={node.color} />
          </g>
        ))}
      </svg>
      <div className="network-labels">
        {nodes.map((node, index) => (
          <div key={node.label} className={`network-tag network-tag-${index + 1}`}>
            <span className="mono network-id" style={{ color: node.color }}>{node.icon}</span>
            <span>{node.label}</span>
          </div>
        ))}
      </div>
      <div className="network-readout mono"><span className="readout-key">FLOW</span><span className="readout-value">6-step inference trace</span></div>
    </div>
  );
}


