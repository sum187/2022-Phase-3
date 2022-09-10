import useMediaQuery from "@mui/material/useMediaQuery";

export default function SimpleMediaQuery() {
  const matches = useMediaQuery("(max-width:200px)");
  const matches1 = useMediaQuery("(max-width:400px)");
  const matches2 = useMediaQuery("(max-width:600px)");
  const matches3 = useMediaQuery("(max-width:800px)");
  const matches4 = useMediaQuery("(min-width:800px)");

  /*1025px — 1200px: Desktops, large screens or anything larger*/
  if (matches===true){
    return 1;
    /*769px — 1024px: Small screens, laptops*/
  } else if (matches1===true){
    return 2;
    /*481px — 768px: iPads, Tablets*/
  } else if (matches2===true){
    return 3;
    /*320px — 480px: Mobile devices*/
  } else if (matches3===true){
    return 5;
    /*320px or less — anything with screen size smaller*/
  } else if (matches4===true){
    return 6;
  }
}