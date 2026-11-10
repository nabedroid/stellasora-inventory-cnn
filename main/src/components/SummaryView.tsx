import React from 'react';
import type { AnalyzedItem } from '@common/types';
import { ItemMasterData } from '@common/services/itemMasterService';

interface SummaryViewProps {
  results: AnalyzedItem[];
  masterData: ItemMasterData[];
}

export const SummaryView: React.FC<SummaryViewProps> = ({ results, masterData }) => {

  const getQty = (name: string) => {
    const found = results.find(r => r.name === name);
    return found?.quantity ? Number(found.quantity) : 0;
  };

  const getIcon = (name: string) => {
    const found = masterData.find(d => d.name === name);
    return found?.iconDataUrl || '';
  };

  const renderCell = (name: string) => {
    const qty = getQty(name);
    const iconUrl = getIcon(name);
    return (
      <td key={name} title={name} style={{ textAlign: 'center', padding: '6px', border: '1px solid var(--border)' }}>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '4px' }}>
          {iconUrl ? (
            <img src={iconUrl} alt={name} style={{ width: '40px', height: '40px', borderRadius: '4px' }} />
          ) : (
            <div style={{ width: '40px', height: '40px', backgroundColor: '#e1e2e6', borderRadius: '4px' }} />
          )}
          <span style={{ fontWeight: 'bold', color: 'var(--primary)', fontSize: '1rem', lineHeight: '1' }}>{qty}</span>
        </div>
      </td>
    );
  };

  const thStyle: React.CSSProperties = { fontSize: '0.8rem', padding: '4px 12px', color: '#57606f', border: '1px solid var(--border)', fontWeight: 'normal', backgroundColor: '#f8f9fa', whiteSpace: 'nowrap' };
  const rowThStyle: React.CSSProperties = { fontSize: '0.9rem', padding: '0 12px', fontWeight: 'bold', color: '#2f3542', border: '1px solid var(--border)', backgroundColor: '#f8f9fa', whiteSpace: 'nowrap' };
  const valTdStyle: React.CSSProperties = { padding: '0 15px', color: '#e55039', fontWeight: 'bold', fontSize: '1.2rem', border: '1px solid var(--border)', whiteSpace: 'nowrap' };

  const cardStyle: React.CSSProperties = {
    backgroundColor: '#ffffff',
    border: '1px solid #e1e2e6',
    borderRadius: '10px',
    padding: '20px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.04)',
    display: 'flex',
    flexDirection: 'column'
  };

  const h3Style: React.CSSProperties = {
    fontSize: '1rem',
    color: '#2f3542',
    margin: '0 0 15px 0',
    borderLeft: '4px solid #4a69bd',
    paddingLeft: '8px'
  };

  return (
    <div style={{ padding: '5px', display: 'flex', flexWrap: 'wrap', gap: '25px', alignItems: 'flex-start' }}>

      <div style={cardStyle}>
        <h3 style={h3Style}>ガイド（巡遊者経験値）</h3>
        <table style={{ borderCollapse: 'collapse', textAlign: 'center', backgroundColor: '#fff' }}>
          <thead>
            <tr>
              <th style={thStyle}></th>
              <th style={thStyle}>入門</th>
              <th style={thStyle}>初級</th>
              <th style={thStyle}>中級</th>
              <th style={thStyle}>上級</th>
              <th style={thStyle}>個分</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style={rowThStyle}>ガイド</td>
              {renderCell('巡遊者入門ガイド')}
              {renderCell('巡遊者初級ガイド')}
              {renderCell('巡遊者中級ガイド')}
              {renderCell('巡遊者上級ガイド')}
              <td style={valTdStyle}>
                {((getQty('巡遊者入門ガイド') * 1000 + getQty('巡遊者初級ガイド') * 5000 + getQty('巡遊者中級ガイド') * 10000 + getQty('巡遊者上級ガイド') * 20000) / 4860060).toFixed(1)}
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <div style={cardStyle}>
        <h3 style={h3Style}>レコード（ロスレコ経験値）</h3>
        <table style={{ borderCollapse: 'collapse', textAlign: 'center', backgroundColor: '#fff' }}>
          <thead>
            <tr>
              <th style={thStyle}></th>
              <th style={thStyle}>ひび</th>
              <th style={thStyle}>量産</th>
              <th style={thStyle}>優雅</th>
              <th style={thStyle}>星空</th>
              <th style={thStyle}>個分</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style={rowThStyle}>レコード</td>
              {renderCell('ひび割れたレコード')}
              {renderCell('量産品のレコード')}
              {renderCell('優雅なレコード')}
              {renderCell('星空のレコード')}
              <td style={valTdStyle}>
                {((getQty('ひび割れたレコード') * 1000 + getQty('量産品のレコード') * 5000 + getQty('優雅なレコード') * 10000 + getQty('星空のレコード') * 20000) / 5180000).toFixed(1)}
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <div style={cardStyle}>
        <h3 style={h3Style}>カセット（スキル強化）</h3>
        <table style={{ borderCollapse: 'collapse', textAlign: 'center', backgroundColor: '#fff' }}>
          <thead>
            <tr>
              <th style={thStyle}></th>
              <th style={thStyle}>初級</th>
              <th style={thStyle}>中級</th>
              <th style={thStyle}>上級</th>
              <th style={thStyle}>スキル分</th>
            </tr>
          </thead>
          <tbody>
            {['音楽', '弾幕', '格闘'].map(type => {
              const equivalent = getQty(`${type}カセット・初級`) + getQty(`${type}カセット・中級`) * 3 + getQty(`${type}カセット・上級`) * 9;
              return (
                <tr key={type}>
                  <td style={rowThStyle}>{type}</td>
                  {renderCell(`${type}カセット・初級`)}
                  {renderCell(`${type}カセット・中級`)}
                  {renderCell(`${type}カセット・上級`)}
                  <td style={valTdStyle}>{(equivalent / 3267).toFixed(1)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div style={cardStyle}>
        <h3 style={h3Style}>首領素材（スキル強化）</h3>
        <table style={{ borderCollapse: 'collapse', textAlign: 'center', backgroundColor: '#fff' }}>
          <thead>
            <tr>
              <th style={thStyle}></th>
              <th style={thStyle}>闇</th>
              <th style={thStyle}>地</th>
              <th style={thStyle}>火</th>
              <th style={thStyle}>風</th>
              <th style={thStyle}>光</th>
              <th style={thStyle}>水</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style={rowThStyle}>部位</td>
              {['罪なる紅花', '森の精霊の牙', '巨神兵のコア', '激怒美人', '翠煌羽の髪飾り', '蒼海の宝木'].map(name => renderCell(name))}
            </tr>
            <tr>
              <td style={rowThStyle}>遺魂</td>
              {['罪花の精魂', '森霊の精魂', '巨神兵の精魂', '風流の精魂', '翠刃の精魂', '蒼海の精魂'].map(name => renderCell(name))}
            </tr>
          </tbody>
        </table>
      </div>

      <div style={cardStyle}>
        <h3 style={h3Style}>モンスター素材（巡遊者昇格）</h3>
        <table style={{ borderCollapse: 'collapse', textAlign: 'center', backgroundColor: '#fff' }}>
          <thead>
            <tr>
              <th style={thStyle}></th>
              <th style={thStyle}>初級</th>
              <th style={thStyle}>中級</th>
              <th style={thStyle}>上級</th>
              <th style={thStyle}>人分</th>
            </tr>
          </thead>
          <tbody>
            {[
              { type: '怪奇ノ霊', low: '楽屋の古い帽子', mid: '道化のフェルトハット', high: 'カーテンコールの礼帽' },
              { type: '癒悦ノ光', low: '陰鬱なる灯心', mid: '凝縮された灯料', high: '永昏の灯核' },
              { type: '空壺ノ王', low: '男爵の報酬', mid: '男爵のとっておき', high: '男爵の恵み' },
            ].map(row => {
              const equivalent = getQty(row.low) + getQty(row.mid) * 3 + getQty(row.high) * 9;
              return (
                <tr key={row.low}>
                  <td style={rowThStyle}>{row.type}</td>
                  {renderCell(row.low)}
                  {renderCell(row.mid)}
                  {renderCell(row.high)}
                  <td style={valTdStyle}>{(equivalent / 1930).toFixed(1)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div style={cardStyle}>
        <h3 style={h3Style}>モンスター素材（ロスレコ昇格）</h3>
        <table style={{ borderCollapse: 'collapse', textAlign: 'center', backgroundColor: '#fff' }}>
          <thead>
            <tr>
              <th style={thStyle}></th>
              <th style={thStyle}>初級</th>
              <th style={thStyle}>中級</th>
              <th style={thStyle}>上級</th>
              <th style={thStyle}>個分</th>
            </tr>
          </thead>
          <tbody>
            {[
              { type: '怪奇ノ霊', low: 'ラビッシュの残滓', mid: '幻歩の遺魂', high: '狂乱の精魂' },
              { type: '癒悦ノ光', low: '薄光の残滓', mid: '蛍火の遺魂', high: '常夜の精魂' },
              { type: '空壺ノ王', low: 'ドボの残滓', mid: 'ルボの遺魂', high: 'ドボの精魂' },
            ].map(row => {
              const equivalent = getQty(row.low) + getQty(row.mid) * 3 + getQty(row.high) * 9;
              return (
                <tr key={row.low}>
                  <td style={rowThStyle}>{row.type}</td>
                  {renderCell(row.low)}
                  {renderCell(row.mid)}
                  {renderCell(row.high)}
                  <td style={valTdStyle}>{(equivalent / 2553).toFixed(1)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div style={cardStyle}>
        <h3 style={h3Style}>徽章</h3>
        <table style={{ borderCollapse: 'collapse', textAlign: 'center', backgroundColor: '#fff' }}>
          <thead>
            <tr>
              <th style={thStyle}></th>
              <th style={thStyle}>70</th>
              <th style={thStyle}>80</th>
              <th style={thStyle}>90</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style={rowThStyle}>徽章</td>
              {['騎士道精神コイン', '信仰の証', '冒険者の証'].map(name => renderCell(name))}
            </tr>
          </tbody>
        </table>
      </div>

      <div style={cardStyle}>
        <h3 style={h3Style}>駒（スキル強化）</h3>
        <table style={{ borderCollapse: 'collapse', textAlign: 'center', backgroundColor: '#fff' }}>
          <thead>
            <tr>
              <th style={thStyle}></th>
              <th style={thStyle}>初級</th>
              <th style={thStyle}>スキル分</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style={rowThStyle}>駒</td>
              {renderCell('テクニックの駒')}
              <td style={valTdStyle}>{(getQty('テクニックの駒') / 2057).toFixed(1)}</td>
            </tr>
          </tbody>
        </table>
      </div>

    </div>
  );
};
