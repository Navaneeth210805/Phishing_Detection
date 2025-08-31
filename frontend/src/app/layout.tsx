import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Phishing Detection System",
  description: "AI-powered phishing detection system for Critical Sector Entities (CSEs)",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased min-h-screen bg-background font-sans`}
      >
        <header className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
          <div className="container mx-auto px-6 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <h1 className="text-xl font-bold">Phishing Detection System</h1>
                <span className="text-sm text-muted-foreground">Critical Sector Entities Monitor</span>
              </div>
              <div className="flex items-center space-x-4">
                <span className="text-sm text-muted-foreground">v1.0.0</span>
              </div>
            </div>
          </div>
        </header>
        <main className="flex-1">
          {children}
        </main>
        <footer className="border-t py-6 px-6">
          <div className="container mx-auto text-center text-sm text-muted-foreground">
            <p>Phishing Detection System - Built for CSE protection and monitoring.</p>
          </div>
        </footer>
      </body>
    </html>
  );
}
