import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, ResponsiveContainer, YAxis, XAxis, CartesianGrid } from 'recharts';
import './App.css';

function App() {
  const [data, setData] = useState(null);
  const [history, setHistory] = useState([]);
  const [scenario, setScenario] = useState('Normal');
  const scenarioRef = React.useRef(scenario);
  
  useEffect(() => {
    scenarioRef.current = scenario;
  }, [scenario]);

  const [connected, setConnected] = useState(false);
  const [alarmPopupVisible, setAlarmPopupVisible] = useState(false);
  const [isLogModalOpen, setIsLogModalOpen] = useState(false);
  const [logFilterType, setLogFilterType] = useState('All');
  const [logStartTime, setLogStartTime] = useState('');
  const [logEndTime, setLogEndTime] = useState('');
  const [logCurrentPage, setLogCurrentPage] = useState(1);
  const logsPerPage = 15;

  const [logs, setLogs] = useState([]);

  useEffect(() => {
    // Tải dữ liệu ban đầu từ DB
    axios.get('http://localhost:8000/api/history')
      .then(res => setHistory(res.data))
      .catch(err => console.error("History fetch error:", err));

    axios.get('http://localhost:8000/api/logs')
      .then(res => setLogs(res.data))
      .catch(err => console.error("Logs fetch error:", err));
  }, []);

  const addLog = (message, type = 'info') => {
    const time = new Date().toLocaleTimeString('en-US', { hour12: false });
    const newLog = { time, message, type };
    
    // Cập nhật UI ngay lập tức
    setLogs(prev => [newLog, ...prev].slice(0, 1000));
    
    // Ghi xuống Backend
    axios.post('http://localhost:8000/api/logs', newLog)
      .catch(err => console.error("Save log error:", err));
  };

  // Derive fake UI inference
  const displayInference = data && data.ai_inference ? { ...data.ai_inference } : null;
  if (displayInference) {
    if (scenario === 'Caution') {
      displayInference.idle = false;
      displayInference.status = 'Yellow';
      displayInference.rca = [
        { feature: 'TP2 (bar)', error: 0.15 },
        { feature: 'LPS (bar)', error: 0.12 },
        { feature: 'H1 (psi)', error: 0.08 }
      ];
    } else if (scenario === 'Warning') {
      displayInference.idle = false;
      displayInference.status = 'Red';
      displayInference.rca = [
        { feature: 'TP2 (bar)', error: 0.45 },
        { feature: 'LPS (bar)', error: 0.38 },
        { feature: 'H1 (psi)', error: 0.25 }
      ];
    } else {
      displayInference.rca = [
        { feature: 'TP2 (bar)', error: 0.012 },
        { feature: 'LPS (bar)', error: 0.009 },
        { feature: 'H1 (psi)', error: 0.005 }
      ];
    }
  }

  // Hardcoded threshold for UI
  // const THRESHOLD = 0.0350;

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/metrics');
    
    ws.onopen = () => { setConnected(true); addLog('Telemetry stream synchronized', 'success'); };
    ws.onclose = () => { setConnected(false); addLog('Telemetry stream disconnected. Reconnecting...', 'error'); };
    ws.onerror = (e) => console.error("WebSocket Error:", e);
    
    ws.onmessage = (event) => {
      try {
        const parsed = JSON.parse(event.data);
        if (parsed && parsed.raw_data) {
          
          // --- FAKE MAE INJECTION ---
          const currentScenario = scenarioRef.current;
          let fakeMae = parsed.ai_inference?.mae_absolute || 0;
          let threshold = parsed.ai_inference?.threshold || 0.0350;
          
          if (currentScenario === 'Caution') {
            // Giả lập MAE nằm trong vùng 80% - 99% threshold
            fakeMae = threshold * 0.85 + (Math.random() * (threshold * 0.14)); 
          } else if (currentScenario === 'Warning') {
            // Giả lập MAE vượt 100% threshold
            fakeMae = threshold * 1.1 + (Math.random() * (threshold * 0.4));
          }
          
          if (parsed.ai_inference) {
             parsed.ai_inference.mae_absolute = fakeMae;
             parsed.ai_inference.mae_percentage = (fakeMae / threshold) * 100;
          }
          // --- END FAKE MAE INJECTION ---

          setData(parsed);
          
          setHistory(prev => {
              const newPoint = {
                  time: new Date().toLocaleTimeString('en-US', { hour12: false }),
                  mae: fakeMae, 
                  ...parsed.raw_data
              };
              const newHist = [...prev, newPoint];
              if (newHist.length > 360) newHist.shift(); // 6 points/min * 60 min = 360 points
              return newHist;
          });
        }
      } catch (err) {}
    };

    return () => ws.close();
  }, []);

  useEffect(() => {
    if (scenario === 'Warning') {
      setAlarmPopupVisible(true);
    }
  }, [scenario]);

  const playBeep = () => {
    try {
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const oscillator = audioCtx.createOscillator();
      const gainNode = audioCtx.createGain();
      
      oscillator.type = 'square';
      oscillator.frequency.setValueAtTime(800, audioCtx.currentTime); // 800Hz beep
      
      gainNode.gain.setValueAtTime(0.1, audioCtx.currentTime); // Volume 10%
      
      oscillator.connect(gainNode);
      gainNode.connect(audioCtx.destination);
      
      oscillator.start();
      setTimeout(() => oscillator.stop(), 200); // 200ms beep duration
    } catch (e) {
      console.log("AudioContext not supported or blocked", e);
    }
  };

  const exportLogsToCSV = () => {
    let csvContent = "data:text/csv;charset=utf-8,Time,Type,Message\n";
    logs.forEach(row => {
      csvContent += `${row.time},${row.type},"${row.message.replace(/"/g, '""')}"\n`;
    });
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `system_logs_${new Date().getTime()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const prevStatusRef = React.useRef(null);
  
  useEffect(() => {
    if (displayInference && displayInference.status) {
      if (prevStatusRef.current !== displayInference.status) {
        if (displayInference.status === 'Red') {
          setAlarmPopupVisible(true);
          playBeep();
          addLog('CRITICAL: AI detected threshold metrics exceeded', 'error');
          if (displayInference.rca && displayInference.rca.length > 0) {
            addLog(`Triggering factor identified: ${displayInference.rca[0].feature}`, 'error');
          }
        } else if (displayInference.status === 'Yellow') {
          addLog('CAUTION: Variance detected in monitoring factors', 'warning');
        } else if (displayInference.status === 'Green' && prevStatusRef.current) {
          addLog('System returned to NORMAL operational state', 'success');
        }
        prevStatusRef.current = displayInference.status;
      }
    }
  }, [displayInference?.status]);

  useEffect(() => {
    const interval = setInterval(() => {
      if (connected) {
         addLog('Routine AI inference cycle stable. Baseline metrics normal.', 'info');
      }
    }, 45000); // Check every 45 seconds
    return () => clearInterval(interval);
  }, [connected]);

  const changeScenario = (newScenario) => {
    // Send 'Normal' to backend to keep models outputting normal raw data
    axios.get(`http://localhost:8000/api/scenario/Normal`)
      .then(() => setScenario(newScenario))
      .catch(err => console.error("Scenario change failed"));
  };

  const getSystemState = () => {
    if (!connected) return { class: 'state-muted', text: 'SYSTEM OFFLINE' };
    if (!displayInference) return { class: 'state-warning', text: 'SYNCING...' };
    
    const inf = displayInference;
    if (inf.idle) return { class: 'state-muted', text: 'IDLE' };
    if (inf.status === 'Red') return { class: 'state-alarm', text: 'ALARM CRITICAL' };
    if (inf.status === 'Yellow') return { class: 'state-warning', text: 'CAUTION ACTIVE' };
    return { class: 'state-normal', text: 'OPERATIONAL' };
  };

  const sysState = getSystemState();
  
  // Lấy giá trị trực tiếp từ Backend đã xử lý
  const mae = data?.ai_inference?.mae_absolute || 0;
  // Lấy ngưỡng động từ AI Engine trả về, nếu chưa có thì mặc định 0.035
  const dynamicThreshold = data?.ai_inference?.threshold || 0.0350; 
  const delta = mae - dynamicThreshold;
  
  // Ưu tiên lấy % từ backend, nếu không có thì dự phòng bằng cách tự tính
  let maePercent = data?.ai_inference?.mae_percentage || (mae / dynamicThreshold) * 100;
  if (isNaN(maePercent) || !connected) maePercent = 0;
  const safePercent = Math.min(100, Math.max(0, maePercent));
  
  const circumference = 2 * Math.PI * 18;
  const strokeDashoffset = circumference - (safePercent / 100) * circumference;
  // Signal colors defined in CSS
  const gaugeColor = maePercent >= 100 ? 'var(--error)' : maePercent >= 80 ? 'var(--secondary)' : 'var(--primary)';

  return (
    <div className="dashboard-container">
      {/* HEADER */}
      <header className="header">
        <div className="logo">METRO AIRFLOW ANOMALY DETECTION</div>
        
        <div className="header-controls">
          <select 
            className="dropdown-select" 
            value={scenario} 
            onChange={(e) => changeScenario(e.target.value)}
          >
            <option value="Normal">NORMAL</option>
            <option value="Caution">CAUTION</option>
            <option value="Warning">WARNING</option>
          </select>
        </div>
      </header>

      {/* MAIN CONTENT */}
      <main className="main-content">
        
        {/* LEFT COLUMN */}
        <div className="left-column">
          
          {/* MAE Panel */}
          <div className="card mae-panel">
            <div className="mae-section mae-gauge-sec">
              <svg viewBox="0 0 36 36" className="circular-chart">
                <path className="circle-bg"
                  d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                />
                <path className="circle"
                  strokeDasharray={`${circumference} ${circumference}`}
                  style={{ strokeDashoffset, stroke: gaugeColor }}
                  d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                />
                <text x="18" y="21" className="percentage">{Math.round(maePercent)}%</text>
                <text x="18" y="26" className="percentage-label">THRESHOLD</text>
              </svg>
            </div>

            <div className="mae-section mae-value-sec" style={{ marginLeft: '0rem', marginRight: '0rem', padding: '1.5rem' }}>
              <div className="label-sm" style={{ marginBottom: '1rem' }}>CURRENT ERROR VALUE</div>
              <div className="display-lg" style={{ color: gaugeColor, marginBottom: '2rem' }}>
                {mae.toFixed(4)}
              </div>
              <div style={{ display: 'flex', width: '100%', gap: '8px'}}>
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', border: '1px solid var(--outline-variant)', background: 'var(--surface-highest)', padding: '0 0.5rem'}}>
                  <span className="label-sm">LIMIT</span>
                  <span className="label-md" style={{ color: gaugeColor }}>{dynamicThreshold.toFixed(4)}</span>
                </div>
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', border: '1px solid var(--outline-variant)', background: 'var(--surface-highest)', padding: '0 0.5rem' }}>
                  <span className="label-sm">DELTA</span>
                  <span className="label-md" style={{ color: delta > 0 ? 'var(--error)' : 'var(--on-surface-variant)' }}>
                    {delta > 0 ? '+' : ''}{delta.toFixed(4)}
                  </span>
                </div>
              </div>
            </div>

            <div className="mae-section mae-trend-sec">
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem' }}>
                <span className="headline-sm" style={{ fontSize: '0.9rem' }}>MAE TREND</span>
                <span className={`label-sm ${sysState.class}`} style={{ display: 'flex', alignItems: 'center', gap: '4px', background: 'var(--surface-low)'}}>
                  <span className="status-pulse" style={{ background: 'var(--surface-low)'}}></span> LIVE
                </span>
              </div>
              <div style={{ flex: 1 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={history} margin={{ top: 5, right: 0, left: 0, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--outline-variant)" vertical={false} />
                    <XAxis dataKey="time" tick={{ fontSize: '0.65rem', fill: 'var(--on-surface-variant)' }} axisLine={{ stroke: 'var(--outline-variant)' }} tickLine={false} minTickGap={30} />
                    <YAxis domain={['auto', 'auto']} hide />
                    <Line type="monotone" dataKey="mae" stroke="var(--primary)" strokeWidth={2} dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* SENSOR TELEMETRY CLUSTER */}
          <div className="card" style={{ backgroundColor: 'var(--surface)', border: 'none', flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
            <div className="card-header" style={{ padding: '0rem 0rem 1rem 0rem', flexShrink: 0 }}>
              <span className="headline-sm">SENSOR TELEMETRY CLUSTER</span>
              {/* <span className="label-sm">NODE: SC-COMP-01</span> */}
            </div>
            <div className="telemetry-grid" style={{ padding: '0rem', overflowY: 'auto', flex: 1, alignContent: 'start', paddingRight: '4px' }}>
              {data && data.raw_data ? Object.keys(data.raw_data).map(key => {
                  if (key === 'timestamp' || key === 'Unnamed: 0') return null;
                  const val = data.raw_data[key];
                  
                  let unit = "";
                  if (key.includes('TP') || key.includes('STAGE')) unit = "BAR";
                  else if (key.includes('H1')) unit = "PSI";
                  else if (key.includes('CURRENT')) unit = "AMP";
                  else if (key.includes('DV_PRESSURE')) unit = "MPa";
                  else if (key.includes('FLOW')) unit = "L/M";
                  else if (key.includes('VIBE')) unit = "g";
                  else if (key.includes('HZ')) unit = "Hz";
                  else if (key.includes('TEMP')) unit = "°C";
                  else if (key.includes('GAL') || key.includes('TANK')) unit = "GAL";
                  else if (key.includes('FILTER') || key.includes('LVL') || key === 'COMP') unit = "%";

                  return (
                    <div key={key} className="telemetry-cell">
                      <span className="label-sm">{key}</span>
                      <div className="tele-val-container">
                        <span className="tele-val">{typeof val === 'number' ? val.toFixed(2) : val}</span>
                        <span className="label-sm">{unit}</span>
                      </div>
                      <div className="tele-chart-box">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={history.filter(p => p[key] !== undefined)}>
                              <YAxis domain={['auto', 'auto']} hide />
                              <Line type="monotone" dataKey={key} stroke="var(--primary)" strokeWidth={2} dot={false} isAnimationActive={false} />
                            </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  );
              }) : (
                 <div style={{ padding: '2rem' }} className="label-md">Awaiting telemetry datastream...</div>
              )}
            </div>
          </div>

        </div>

        {/* RIGHT COLUMN */}
        <div className="right-column">

          {/* NEW SYSTEM STATUS CARD */}
          {(() => {
                const PauseIcon = () => 
                  <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M18.36 6.64a9 9 0 1 1-12.73 0" />
                    <line x1="12" y1="2" x2="12" y2="12" />
                  </svg>;
                const CheckIcon = () => 
                  <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                      <path d="M9 12l2 2 4-4" />
                  </svg>;
                const AlertTriangleIcon = () => 
                  <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
                      <line x1="12" y1="9" x2="12" y2="13" />
                      <circle cx="12" cy="17" r="0.5" fill="currentColor" />
                  </svg>;                
                const AlertOctagonIcon = () => 
                  <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                    <polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2" />
                    <line x1="12" y1="8" x2="12" y2="12" />
                    <line x1="12" y1="16" x2="12.01" y2="16" />
                  </svg>;                
                let statusText = "SYNCING...";
                let cardBg = "var(--surface-low)";
                let cardText = "var(--on-surface)";
                let Icon = PauseIcon;
                let isAlarm = false;
                
                if (!connected) {
                  statusText = "SYSTEM OFFLINE";
                  cardBg = "var(--surface-low)";
                  cardText = "var(--on-surface-variant)";
                  Icon = AlertTriangleIcon;
                } else if (displayInference) {
                  const inf = displayInference;
                  if (inf.idle) {
                    statusText = "INACTIVE";
                    cardBg = "#a7a7a7ff"; // xám trắng
                    cardText = "#000000"; // đen
                    Icon = PauseIcon;
                  } else {
                    if (inf.status === 'Green') {
                      statusText = "NORMAL";
                      cardBg = "var(--primary)"; // xanh
                      cardText = "#000000"; // đen
                      Icon = CheckIcon;
                    } else if (inf.status === 'Yellow') {
                      statusText = "CAUTION";
                      cardBg = "var(--secondary)"; // vàng
                      cardText = "#000000"; // đen
                      Icon = AlertTriangleIcon;
                    } else if (inf.status === 'Red') {
                      statusText = "WARNING";
                      cardBg = "#d32f2f"; // đỏ đậm
                      cardText = "#ffffff"; // trắng
                      Icon = AlertOctagonIcon;
                      isAlarm = true;
                    }
                  }
                }

                return (
                  <div className={`card ${isAlarm ? 'card-flash-red' : ''}`} style={{ backgroundColor: isAlarm ? undefined : cardBg, color: cardText, transition: 'all 0.3s ease', border: 'none' }}>
                    <div className="card-header" style={{ paddingBottom: '0rem' }}>
                      <span className="headline-sm" style={{ color: isAlarm ? '#ffffff' : cardText, opacity: 0.8, fontSize: '0.75rem' }}>SYSTEM STATUS</span>
                    </div>
                    <div style={{ padding: '1.5rem 1.5rem', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                        <div style={{ display: 'flex', animation: isAlarm ? 'pulseError 1.5s infinite' : 'none' }}>
                          <Icon />
                        </div>
                        <span className="display-lg" style={{ color: cardText, fontSize: '1.5rem', letterSpacing: '1px' }}>
                          {statusText}
                        </span>
                      </div>
                    </div>
                  </div>
                );
          })()}
          
          <div className="card rca-card bg-state-normal">
            {/* <div className="rca-header-solid">
              <span className="headline-sm">{sysState.text}</span>
            </div> */}

            <div className="rca-body">
                <div style={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0, overflow: 'hidden' }}>
                  
                  {/* NODE METADATA */}
                  <div style={{ marginBottom: '1.5rem' }}>
                    <div className="rca-header-solid" style={{ marginBottom: '0.75rem', borderBottom: '1px solid var(--outline-variant)', padding: '0', paddingBottom: '0.75rem' }}>
                      <span className="headline-sm">SYSTEM INFORMATION</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.4rem' }}>
                      <span className="label-sm">VEHICLE UNIT</span>
                      <span className="label-md" style={{ color: 'var(--on-surface)' }}>MP-Series 2800</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.4rem' }}>
                      <span className="label-sm">TRAIN ID</span>
                      <span className="label-md" style={{ color: 'var(--on-surface)' }}>TRN-03-A</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.4rem' }}>
                      <span className="label-sm">ACTIVE LINE</span>
                      <span className="label-md" style={{ color: 'var(--on-surface)' }}>Linha D (Amarela)</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span className="label-sm">COMPONENT</span>
                      <span className="label-md" style={{ color: 'var(--on-surface)' }}>Reciprocating Air Compressor</span>
                    </div>
                  </div>

                  {/* ACTIVITY LOG */}
                  <div style={{ display: 'flex', flexDirection: 'column', flex: 1, marginBottom: '0rem', minHeight: 0 }}>
                    <div className="rca-header-solid" style={{ marginBottom: '0.75rem', borderBottom: '1px solid var(--outline-variant)', padding: '0 0 0.75rem 0', flexShrink: 0, alignItems: 'center' }}>
                      <span className="headline-sm">ACTIVITY LOG</span>
                      <button className="label-sm" style={{ background: 'none', border: '1px solid var(--outline-variant)', color: 'var(--on-surface)', cursor: 'pointer', padding: '2px 8px', borderRadius: '2px' }} onClick={() => setIsLogModalOpen(true)}>VIEW FULL LOGS</button>
                    </div>
                    <div className="activity-log-container" style={{
                      background: 'var(--surface-highest)', 
                      padding: '0.75rem', 
                      borderRadius: '0.125rem',
                      display: 'flex', 
                      flexDirection: 'column', 
                      gap: '0.5rem',
                      fontFamily: 'var(--font-sans)',
                      fontSize: '0.75rem',
                      flex: 1,
                      overflowY: 'auto'
                    }}>
                      {logs.slice(0, 50).map((log, idx) => {
                         let color = 'var(--on-surface-variant)';
                         if (log.type === 'error') color = 'var(--error)';
                         if (log.type === 'warning') color = 'var(--secondary)';
                         if (log.type === 'success') color = 'var(--primary)';
                         if (log.type === 'info') color = 'var(--on-surface)';
                         
                         return (
                           <div key={idx} style={{ display: 'flex', gap: '0.5rem', opacity: log.type === 'info' ? 0.8 : 1 }}>
                             <span style={{ color: color, fontWeight: 'bold', flexShrink: 0 }}>[{log.time}]</span> 
                             <span style={{ color: log.type === 'info' ? 'var(--on-surface-variant)' : 'var(--on-surface)' }}>{log.message}</span>
                           </div>
                         );
                      })}
                      {logs.length === 0 && <div style={{ color: 'var(--on-surface-variant)' }}>Awaiting system events...</div>}
                    </div>
                  </div>
                </div>

              <button className={`btn-primary ${displayInference?.status === 'Red' ? 'btn-alarm' : (displayInference?.status === 'Yellow' ? 'btn-caution' : '')}`} onClick={() => setAlarmPopupVisible(true)}>Index Analysis</button>
            </div>
          </div>

        </div>
      </main>

      {/* FOOTER */}
      <footer className="footer">
        <div style={{ display: 'flex', gap: '2rem' }}>
          <span className="label-sm">UPTIME: 142:12:04</span>
          <span className="label-sm">LATENCY: 14ms</span>
        </div>
        <div style={{ display: 'flex', gap: '1.5rem' }}>
          <span className={`label-sm state-normal`} style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <span className="status-pulse"></span> CORE
          </span>
          <span className={`label-sm ${sysState.class}`} style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <span className="status-pulse"></span> {data && data.ai_inference && !data.ai_inference.idle ? 'ENGINE' : 'IDLE'}
          </span>
        </div>
      </footer>

      {/* ALARM POPUP NOTIFICATION (Glass/Gradient Rule) */}
      {alarmPopupVisible && displayInference && displayInference.rca && (
        <div className="alarm-overlay">
          <div className={`alarm-modal ${displayInference.status === 'Red' ? 'alarm-modal-flashing' : ''}`} style={{ borderLeft: `8px solid ${displayInference.status === 'Red' ? '#d32f2f' : (displayInference.status === 'Yellow' ? 'var(--secondary)' : 'var(--primary)')}` }}>
            
            {/* Headline Chuyên nghiệp */}
            <div className="headline-sm" style={{ 
              color: displayInference.status === 'Red' ? '#d32f2f' : (displayInference.status === 'Yellow' ? 'var(--secondary)' : 'var(--primary)'), 
              marginBottom: '0.5rem', 
              fontSize: '1.25rem',
              fontWeight: '800',
              letterSpacing: '1px'
            }}>
              {displayInference.status === 'Red' ? '⚠️ CRITICAL FAULT ISOLATION' : (displayInference.status === 'Yellow' ? '🔔 PREDICTIVE CAUTION' : '✅ SYSTEM DIAGNOSTIC')}
            </div>

            {/* Thông báo ngữ cảnh kỹ thuật */}
            <div className="label-md" style={{ marginBottom: '1.5rem', color: 'var(--on-surface-variant)', fontStyle: 'italic' }}>
              {displayInference.status === 'Red' 
                ? 'Pneumatic divergence detected. High probability of air leak in ASU Unit TRN-03-A. Inspection required.' 
                : (displayInference.status === 'Yellow' 
                   ? 'Minor variance in duty cycle detected.' 
                   : 'System performing within nominal 4.1-sigma parameters.')}
            </div>

            {/* Phân tích nguyên nhân gốc rễ (RCA) - CHỈ HIỂN THỊ KHI KHÔNG PHẢI GREEN */}
            {displayInference.status !== 'Green' && (
              <div style={{ background: 'var(--surface-high)', padding: '1.25rem', marginBottom: '1.5rem', borderRadius: '4px' }}>
                <div className="label-sm" style={{ marginBottom: '1rem', borderBottom: '1px solid var(--outline-variant)', paddingBottom: '0.5rem' }}>
                  ROOT CAUSE ANALYSIS (TOP CONTRIBUTIONS)
                </div>
                
                {displayInference.rca.slice(0, 3).map((cause, idx) => {
                  // Giả lập tính % đóng góp dựa trên giá trị error
                  const totalError = displayInference.rca.reduce((acc, curr) => acc + curr.error, 0);
                  const contribution = ((cause.error / totalError) * 100).toFixed(1);
                  
                  return (
                    <div key={idx} style={{ marginBottom: '1rem' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                        <span className="label-md" style={{ fontWeight: idx === 0 ? '700' : '400' }}>
                          {idx + 1}. {cause.feature}
                        </span>
                        <span className="label-md" style={{ color: idx === 0 ? '#d32f2f' : 'inherit' }}>{contribution}%</span>
                      </div>
                      {/* Thanh Progress Bar trực quan */}
                      <div style={{ width: '100%', height: '4px', background: 'var(--surface-low)', borderRadius: '2px', overflow: 'hidden' }}>
                        <div style={{ 
                          width: `${contribution}%`, 
                          height: '100%', 
                          background: idx === 0 ? '#d32f2f' : 'var(--primary)',
                          transition: 'width 1s ease-in-out'
                        }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {/* Nút hành động */}
            <div style={{ display: 'flex', gap: '1rem' }}>
              <button className="btn-primary" style={{ flex: 1, textTransform: 'uppercase', fontWeight: 'bold' }} onClick={() => setAlarmPopupVisible(false)}>
                {displayInference.status === 'Green' ? 'CLOSE DIAGNOSTIC' : 'ACKNOWLEDGE & REPORT TO DEPOT'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* LOG MODAL POPUP */}
      {isLogModalOpen && (
        <div className="alarm-overlay">
          <div className="log-modal">
            <div className="log-modal-header">
              <span className="headline-sm">SYSTEM EVENT LOGS</span>
              <button className="btn-primary" style={{ padding: '0.5rem 1rem', fontSize: '0.7rem' }} onClick={() => setIsLogModalOpen(false)}>CLOSE</button>
            </div>
            <div className="log-modal-filters">
              <select className="log-filter-select" value={logFilterType} onChange={e => { setLogFilterType(e.target.value); setLogCurrentPage(1); }}>
                <option value="All">All Types</option>
                <option value="info">Info</option>
                <option value="warning">Warning</option>
                <option value="error">Error</option>
                <option value="success">Success</option>
              </select>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span className="label-sm">FROM</span>
                <input 
                  type="time"
                  lang="en-US"
                  className="log-filter-input" 
                  value={logStartTime} 
                  onChange={e => { setLogStartTime(e.target.value); setLogCurrentPage(1); }} 
                />
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span className="label-sm">TO</span>
                <input 
                  type="time"
                  lang="en-US"
                  className="log-filter-input" 
                  value={logEndTime} 
                  onChange={e => { setLogEndTime(e.target.value); setLogCurrentPage(1); }} 
                />
              </div>
              <button className="btn-primary" style={{ padding: '0.5rem 1rem', fontSize: '0.7rem', marginLeft: 'auto', background: 'var(--surface-high)', color: 'var(--on-surface)' }} onClick={exportLogsToCSV}>
                EXPORT TO CSV
              </button>
            </div>
            {(() => {
              const filtered = logs.filter(log => {
                if (logFilterType !== 'All' && log.type !== logFilterType) return false;
                if (logStartTime && log.time < logStartTime) return false;
                if (logEndTime && log.time > logEndTime) return false;
                return true;
              });
              const totalPages = Math.ceil(filtered.length / logsPerPage);
              const startIndex = (logCurrentPage - 1) * logsPerPage;
              const currentLogs = filtered.slice(startIndex, startIndex + logsPerPage);
              
              return (
                <>
                  <div className="log-modal-body">
                    {currentLogs.map((log, idx) => {
                      let color = 'var(--on-surface-variant)';
                      if (log.type === 'error') color = 'var(--error)';
                      if (log.type === 'warning') color = 'var(--secondary)';
                      if (log.type === 'success') color = 'var(--primary)';
                      if (log.type === 'info') color = 'var(--on-surface)';
                      return (
                        <div key={idx} className="log-item">
                          <span style={{ color: color, fontWeight: 'bold', minWidth: '80px' }}>[{log.time}]</span>
                          <span style={{ color: log.type === 'info' ? 'var(--on-surface-variant)' : 'var(--on-surface)' }}>{log.message}</span>
                        </div>
                      );
                    })}
                    {filtered.length === 0 && <div style={{ color: 'var(--on-surface-variant)', padding: '1rem' }}>No logs available.</div>}
                  </div>
                  {totalPages > 1 && (
                    <div className="log-pagination">
                       <button className="btn-pagination" disabled={logCurrentPage === 1} onClick={() => setLogCurrentPage(p => p - 1)}>PREV</button>
                       <span className="label-sm">PAGE {logCurrentPage} OF {totalPages}</span>
                       <button className="btn-pagination" disabled={logCurrentPage === totalPages} onClick={() => setLogCurrentPage(p => p + 1)}>NEXT</button>
                    </div>
                  )}
                </>
              );
            })()}
          </div>
        </div>
      )}

    </div>
  );
}

export default App;
