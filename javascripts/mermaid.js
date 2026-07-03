window.addEventListener("load", () => {
  if (!window.mermaid) return
  window.mermaid.initialize({
    startOnLoad: true,
    theme: document.body.dataset.mdColorScheme === "slate" ? "dark" : "default",
    securityLevel: "loose",
    flowchart: { useMaxWidth: true },
    themeVariables: {
      primaryColor: "#8b5cf6",
      primaryTextColor: "#f8fafc",
      primaryBorderColor: "#6d28d9",
      lineColor: "#6366f1",
      secondaryColor: "#ec4899",
      tertiaryColor: "#0f172a"
    }
  })
})
