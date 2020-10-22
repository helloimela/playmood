import React from 'react';
import './App.scss';

export default class App extends React.Component {

  constructor() {
    super();
  }

  render() {
    return (
      <div className="App">
        <h1>Playmood</h1>
        <div className='emojis'></div>
      </div>
    );
  }
}
