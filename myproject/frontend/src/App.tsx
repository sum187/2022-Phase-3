import axios from "axios";
import { useState, useEffect } from 'react';
import * as React from 'react';
import './App.css';
import { Box, Button, Grid, Paper, Skeleton,ImageList,ImageListItem,ImageListItemBar,TextField,Typography} from "@mui/material";
import SearchIcon from "@mui/icons-material/Search";
import Pagination from '@mui/material/Pagination';
import Stack from '@mui/material/Stack';
import Demo from './loading';
import { StyledEngineProvider } from '@mui/material/styles';




function App() {
  const [movieName, setMovieName] = useState<undefined | any>('');
  const [name, setName] = useState<undefined | any>('');
  const [MovieInfo, setMovieInfo] = useState<undefined | any>(undefined);
  const [page, setPage] = React.useState(1);
  const [pageResult, setSearchResult] = useState(1);
  const handleChange = (event: React.ChangeEvent<unknown>, value: number) => {
    setPage(value);
    setSearchResult(value);
  };

  // url and key 
  const Movie_BASE_URL = "https://www.omdbapi.com";
  //const key = "938b78ef";
  //const key = "8059c2e4";
  const key = "52776e8"
  const noImage = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/65/No-Image-Placeholder.svg/330px-No-Image-Placeholder.svg.png?20200912122019"

  //keep track of screen size
  const [windowSize, setWindowSize] = useState(getWindowSize());
  const [col, setCol] = useState(getColumn(windowSize.innerWidth));
  useEffect(() => {
    function handleWindowResize() {
      setWindowSize(getWindowSize());
    }
    window.addEventListener('resize', handleWindowResize);
    return () => {
      window.removeEventListener('resize', handleWindowResize);
      setCol(getColumn(windowSize.innerWidth));
    };
  }, []);

  // keep track of select page
  useEffect(() => {
    search(page)
  }, [pageResult]);

  
  // loading while waiting for the result
  const [loading, setLoading] = useState(false);
  

  



  /*return(
    <React.StrictMode>
      <StyledEngineProvider injectFirst>
        <Demo />
      </StyledEngineProvider>
    </React.StrictMode>
  )*/



  return (
    <div>
      <div className="search-field">
        <h1>Movie Search</h1 >
        <div style={{ display: "flex", justifyContent: "center" }}>

          <TextField
            id="search-bar"
            className="text"
            value={movieName}
            onChange={(prop) => {
              setMovieName(prop.target.value);
            }}
            label="Enter the moive name..."
            variant="outlined"
            placeholder="Search..."
            size="medium"
          />
         
          <Button
            onClick={() => {
              // movie name not entered
              if (movieName===''){ 
                setName('')
              // new movie name is searched
              } else if (movieName!==name) {
                search(page),
                setPage(1),
                setName(movieName)
              // same movie name is searched but not at page 1
              } else if (movieName===name&&page!==1){
                setPage(1)
                search(1)
              }
            }}
          >
            <SearchIcon style={{ fill: "blue" }} />
            Search
          </Button>
        </div>
      </div>
      {MovieInfo === undefined? (
        <div></div>
      ) : (MovieInfo.Response === "False") && (name!=='') ? (
      <div>
          <Paper sx={{ backgroundColor: "#E0FFFF" }}>
            <Grid
              container
              direction="row"
              spacing={0}
              sx={{justifyContent: "center",}}>
              <Grid item>
                <Box>
                <h1>Moive not found</h1>
                  <Skeleton width={300} height={300} />
                </Box>
              </Grid>    
            </Grid>
          </Paper> 
        </div> 
      ) : MovieInfo.Response === "True"? (
        <div className='App'>
          {loading ? <React.StrictMode>
                      <StyledEngineProvider injectFirst>
                        <Demo />
                      </StyledEngineProvider>
                    </React.StrictMode>: 
          <Paper sx={{ backgroundColor: "#191a1a" }}>
            <div
              className='App'
              id="movie-result"
              style={{
                maxWidth: "85%",
                margin: "0 auto",
                padding: "2.5vw 2.5vw 2.5vw 2.5vw",
              }}
            >
              <Paper sx={{ backgroundColor: "#030303" }}>
                <ImageList variant="standard" cols={getColumn(windowSize.innerWidth)} gap={12}>
                  {MovieInfo.Search.map((search:any) => (       
                    <ImageListItem >
                      <img
                        src={`${validLink(search.Poster)}?w=248&fit=crop&auto=format`}
                        srcSet={`${validLink(search.Poster)}?w=248&fit=crop&auto=format&dpr=2 2x`}
                        alt={search.Title}
                        style={{ width: imageSize(windowSize.innerWidth,'width'),
                           height: imageSize(windowSize.innerWidth,'height')}}
                        loading="lazy"
                      />  
                      <ImageListItemBar 
                        title={search.Title} 
                        subtitle={search.Director}
                        />
                    </ImageListItem>
                  ))}
                </ImageList>
              </Paper>
              <Grid
                container
                direction="row"
                spacing={0}
                sx={{
                  justifyContent: "center",
                }}>
                <Paper sx={{ backgroundColor: "#a2a8a5" }}>
                  <div
                    id="movie-result"
                    style={{
                      maxWidth: "100%",
                      margin: "0 auto",
                      //padding: "100px 10px 0px 10px",
                    }}
                  >
                    <Stack spacing={2}>
                      <Typography>Page: {page}</Typography>
                      <Pagination count={
                        Math.ceil(parseInt(MovieInfo.totalResults)/30)} 
                        variant="outlined"
                        shape="rounded"
                        page={page} 
                        onChange={handleChange} 
                        />
                    </Stack>
                  </div>
                </Paper>
              </Grid>
            </div>
          </Paper>
          }
        </div>
      ):(
        <div></div>
      )}   
    </div>
  );

  
  async function search(page:any) {
    setLoading(true)
    console.log(name)
    console.log(movieName)
    console.log(MovieInfo)
    console.log(page)
    console.log(pageResult)
    console.log('loading')
    // get response for first time
    const res = await axios.
    get(Movie_BASE_URL + "/?s=" + movieName + "&apikey=" + key + "&page=" +page); 
    
    if (movieName===''){

    // check response 
    } else if (res.data.Response!=='True'){
      console.log('there was no search');
      setMovieInfo(res.data);
    } else {
    var value=0 

    // making sure page number is not exceed
    const maxPage=Math.ceil(res.data.totalResults/10)
    var endPage=page*3
    if (endPage>maxPage){
      endPage=maxPage
    }

    // loop to call multiple responses (each response has maximum of 10 search results)
      for (let i = page*3-2; i < endPage+1; i++) {

        // get response
        var response = await axios.
          get(Movie_BASE_URL + "/?s=" + movieName + "&apikey=" + key + "&page=" +i.toString());

        // loop to combine responses
        for (let j=0; j<response.data.Search.length;j++){
          res.data.Search[value]=response.data.Search[j]
          value+=1;
        }
      }
      setMovieInfo(res.data);
    }
    console.log('finish')
    setLoading(false)
}  
  
  // if invalid get image that says 'no image'
  function validLink(value: any) {
    if ( value === "N/A") {
      return noImage;
    } else {
      return value;
    }
  }

  function getWindowSize() {
    const {innerWidth, innerHeight} = window;
    return {innerWidth, innerHeight};
  }
  
  // get number of image in the column depending on screen size
  function getColumn(width:any) {
    if (width<320) {
      return 1;
    } else if (width>=320&&width<=480) {
      return 2;
    } else if (width>480&&width<=768) {
      return 3;
    } else if (width>=768&&width<=1024) {
      return 5;
    } else if (width>=1024) {
      return 6;
    }}

  // get image size  
  function imageSize(length:number,string:string) {
    const col=getColumn(window.innerWidth)
    if (col!==undefined&&string==='width'){
      return length/col*0.7;
    } else if (col!==undefined&&string==='height'){
      return length/col*0.7*1.412
    }
  }
}

export default App;
