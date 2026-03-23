import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Mycetoma AI Diagnostics",
  description:
    "AI-powered histopathology analysis for Mycetoma diagnosis — classification, localization, and clinical explanation.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="antialiased min-h-screen">{children}</body>
    </html>
  );
}
