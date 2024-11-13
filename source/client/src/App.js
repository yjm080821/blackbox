import { useState } from "react";
import Header from "./pages/Header";
import Main from "./pages/Main";

function App() {
  const [page, setPage] = useState(0);

  const handlePage = (num, isMobile) => {
    if (isMobile) {
      setPage(page + num);
    } else {
      setPage(num);
    }
  };

  return (
    <div className="App">
      <Header handlePage={handlePage} />
      <div>
        <Main />
        {page}
      </div>
    </div>
  );
}

export default App;
